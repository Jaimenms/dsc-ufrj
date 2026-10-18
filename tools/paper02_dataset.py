"""Massa de dados do artigo 02: do hipercubo das entradas aos marcadores para o substituto (plano do artigo 02).

Entradas (5): P_res_bar, PI, GOR, P_gas_bar, choke_opening, sobre a geometria com gas lift do artigo 01
(`generators/offshore.py` com `gas_lift=true`). Desenho de Sobol embaralhado, agrupado em cadeias de vizinhos
(k-means e ordem do vizinho mais próximo); cada cadeia corre em sequência num processo: o primeiro ponto da partida
bloqueada, os seguintes com arranque quente a partir do estado final do anterior. Cada corrida usa a paragem adaptativa
com captura de ciclos (061): pára quando P na cabeça do poço e q_L no topo do riser assentam (estacionário ou ciclo
repetido) e continua N_c ciclos, que são a janela de captura dos marcadores.

Dentro de cada cadeia a sequência é a do dono: sorteia-se o primeiro ponto e segue-se o caminho mais curto por todos
os outros (exato até 10 pontos), com as entradas reescalonadas a [−1, +1]. Ao acabar uma ronda, `run --rounds` sorteia
outros tantos pontos (a continuação da mesma sequência de Sobol) e recomeça num ponto sorteado da ronda nova com a
partida bloqueada.

Tudo é retomável e nada se perde: o desenho é um ficheiro (`design.json`, só cresce: nunca se substitui quando há
corridas), cada corrida é um `.npz` gravado atomicamente (ou um `.failed.json`), a ferramenta nunca apaga corridas, e
uma cadeia retoma do último ponto concluído; um ficheiro `STOP` na pasta pára as cadeias depois da corrida em curso
(paragem quente) e impede rondas novas; matar os processos perde só a corrida em curso; `markers` só lê. Os marcadores (média, mínimo, máximo, f_min, f_max de
P, alpha_G, q_L e q_G em oito pontos característicos, mais o gás injetado e a classe) vão para `dataset.csv` com o
dicionário das colunas em `dataset.json`.

Rondas em pastas datadas (pedido do dono): `runs/paper02/rounds/rounds.json` é a raiz (semente, faixas, configuração,
lista de rondas) e cada ronda vive em `runs/paper02/rounds/YYYYMMDD/` (sufixo _2, _3… no mesmo dia) com o seu
`design.json`, as corridas e a sua `dataset.csv`; a raiz recebe a tabela de todas as rondas. Omissão: 25 pontos por
ronda, uma cadeia por ronda. `--flat` mantém o modo de uma só pasta.

    uv run python tools/paper02_dataset.py design --out runs/paper02/rounds --seed 1        # raiz + ronda 0 (25 pontos)
    uv run python tools/paper02_dataset.py run --out runs/paper02/rounds --rounds -1        # continuamente, até STOP
    uv run python tools/paper02_dataset.py markers --out runs/paper02/rounds               # tabelas por ronda e total
    uv run python tools/paper02_dataset.py status --out runs/paper02/rounds
"""

from __future__ import annotations

import argparse
import csv
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
    distâncias do caminho mais curto, o centro da distribuição e os grupos."""
    lo = np.array([ranges[k][0] for k in INPUTS])
    hi = np.array([ranges[k][1] for k in INPUTS])
    return 2.0 * (np.asarray(X, float) - lo) / (hi - lo) - 1.0


def sobol_design(n, ranges, seed, skip=0):
    """`n` pontos de Sobol embaralhado nas faixas (matriz n × 5 na ordem de INPUTS), a partir do ponto `skip` da
    sequência: rondas sucessivas da mesma semente continuam a mesma sequência e, juntas, são um só desenho."""
    import warnings

    from scipy.stats import qmc

    gen = qmc.Sobol(d=len(INPUTS), scramble=True, seed=seed)
    if skip:
        gen.fast_forward(int(skip))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # n fora das potências de 2 é deliberado (rondas)
        u = gen.random(int(n))
    lo = np.array([ranges[k][0] for k in INPUTS])
    hi = np.array([ranges[k][1] for k in INPUTS])
    return lo + u * (hi - lo)


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


def make_chains(X, ranges, n_chains, seed, start="random"):
    """Agrupa os pontos em `n_chains` cadeias de vizinhos (k-means nas entradas reescalonadas a [−1, 1]) e, em cada
    cadeia, sorteia o primeiro ponto (`start = "random"`, pedido do dono; `"centre"` = o mais perto do centro do
    grupo) e segue o caminho mais curto por todos os outros (`shortest_path_from`). Devolve (cadeia, ordem) por
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
        if start == "centre":
            centre = U[idx].mean(axis=0)
            first = int(np.argmin(np.linalg.norm(U[idx] - centre, axis=1)))
        else:  # pedido do dono: o primeiro ponto é sorteado antes de se determinar o caminho
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
        signal={"edge": "1", "node": 1, "var": "P", "amp_tol": AMP_TOL["P"]},
        signals=[{"edge": "1", "node": -1, "var": "qL", "amp_tol": AMP_TOL["qL"]}],
    )


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
            X, ranges, chains, int(design["seed"]) + int(design["n"]), start=design.get("start", "random")
        )
    c0 = 1 + max((p["chain"] for p in design["points"]), default=-1)
    rnd = 1 + max((p.get("round", 0) for p in design["points"]), default=-1)
    i0 = int(design["n"])
    width = max(len(str(i0 + n - 1)), 3)
    new = [
        dict(
            id=f"p{i0 + i:0{width}d}",
            x={k: float(X[i, j]) for j, k in enumerate(INPUTS)},
            chain=int(c0 + chain[i]),
            order=int(order[i]),
            round=int(rnd),
        )
        for i in range(n)
    ]
    design["points"].extend(new)
    design["n"] = i0 + n
    design["rounds"] = rnd + 1
    return new


ROUNDS_FILE = "rounds.json"  # raiz das rondas (pedido do dono): uma pasta datada YYYYMMDD por ronda


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
    """Pasta da ronda: a data de hoje (YYYYMMDD), com sufixo _2, _3… se já houver uma ronda de hoje."""
    name = time.strftime("%Y%m%d")
    d, k = Path(root) / name, 1
    while d.exists():
        k += 1
        d = Path(root) / f"{name}_{k}"
    return d


def create_round(root, master, n=None, chains=1):
    """Ronda nova numa pasta datada: os próximos `n` pontos da sequência de Sobol do conjunto (primeiro ponto
    sorteado, caminho mais curto), com um `design.json` próprio; atualiza `rounds.json`. Devolve a pasta."""
    ranges = {k: tuple(v) for k, v in master["ranges"].items()}
    n = int(n or master["round_size"])
    skip = int(master["n"])
    X = sobol_design(n, ranges, int(master["seed"]), skip=skip)
    chain, order = make_chains(X, ranges, chains, int(master["seed"]) + skip, start=master.get("start", "random"))
    rnd = len(master["rounds"])
    width = max(len(str(skip + n - 1)), 3)
    points = [
        dict(
            id=f"p{skip + i:0{width}d}",
            x={k: float(X[i, j]) for j, k in enumerate(INPUTS)},
            chain=int(chain[i]),
            order=int(order[i]),
            round=rnd,
        )
        for i in range(n)
    ]
    design = {k: v for k, v in master.items() if k != "rounds"}
    design.update(n=n, round_size=n, rounds=1, round=rnd, sobol_skip=skip, points=points)
    d = new_round_dir(root)
    d.mkdir(parents=True, exist_ok=False)
    save_design(d, design)
    master["rounds"].append(dict(folder=d.name, n=n, skip=skip, created=time.strftime("%Y-%m-%d %H:%M")))
    master["n"] = skip + n
    save_rounds(root, master)
    return d


def cmd_design(a):
    out = Path(a.out)
    if not a.flat:
        # omissão: raiz de rondas, cada ronda na sua pasta datada
        if is_rounds_root(out):
            if not a.append:
                raise SystemExit(f"{rounds_path(out)} já existe (--append para abrir uma ronda nova)")
            master = load_rounds(out)
            d = create_round(out, master, n=a.n, chains=a.chains)
            print(f"ronda {len(master['rounds']) - 1}: {d} ({master['rounds'][-1]['n']} pontos)")
            return
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} existe e não está vazia: um conjunto nunca se substitui")
        ranges = parse_ranges(a.range)
        master = dict(
            schema="paper02-rounds/1",
            created=time.strftime("%Y-%m-%d %H:%M"),
            seed=int(a.seed),
            n=0,
            round_size=int(a.n or 25),
            start=a.start,
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
        pts = load_design(d)["points"]
        tour = [p["id"] for p in sorted(pts, key=lambda p: p["order"])]
        print(f"{rounds_path(out)}: ronda 0 em {d} com {len(pts)} pontos (semente {a.seed})")
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
        n_runs = len(list(out.glob("p*.npz")))
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
        chain, order = make_chains(X, ranges, a.chains, a.seed, start=a.start)
    width = max(len(str(X.shape[0] - 1)), 3)
    points = [
        dict(
            id=f"p{i:0{width}d}",
            x={k: float(X[i, j]) for j, k in enumerate(INPUTS)},
            chain=int(chain[i]),
            order=int(order[i]),
            round=0,
        )
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
def run_path(out, pid):
    return Path(out) / f"{pid}.npz"


def fail_path(out, pid):
    return Path(out) / f"{pid}.failed.json"


def point_status(out, pid):
    if run_path(out, pid).exists():
        return "done"
    if fail_path(out, pid).exists():
        return "failed"
    return "pending"


def _settle_for(design, start):
    s = dict(design["settle"])
    t_min_warm = s.pop("t_min_warm", None)
    if start == "warm" and t_min_warm is not None:
        s["t_min"] = float(t_min_warm)
    return s


def _node_index(case, edge, node):
    """`node` negativo conta do fim (−1 = última célula) no esquema do duto `edge`."""
    if node >= 0:
        return node
    e = next(x for x in case["network"]["edges"] if x["id"] == edge)
    n_cells = sum(int(sg["n"]) for sg in e["geometry"]["segments"])
    return n_cells + 1 + node


def build_case(design, point, out, start, X_prev=None, prev_id=None):
    """Escreve o caso de um ponto (gerador do offshore com gas lift, paragem adaptativa com captura, orçamento);
    partida bloqueada ou arranque quente a partir de `X_prev`. Devolve o caminho do JSON."""
    from generators.offshore import offshore_case, write_case
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
    cases_dir = Path(out) / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    path = cases_dir / f"{point['id']}.json"
    if start == "blocked":
        write_case(case, path, ic="blocked")
        return path
    net = case_from_json(case)
    model = build_model(net)
    model.state_to_case_ic(np.asarray(X_prev, float), 0.0, exact=True)
    net.initial = "edges"
    net.meta = dict(net.meta or {}, initial_guess=dict(mode="warm", from_point=prev_id))
    case_to_json(net, str(path), ic_file=f"{path.stem}.ic.npz")
    return path


def _final_state(out, pid):
    from mfsim.runs import load_run

    run = load_run(str(run_path(out, pid)))
    return np.asarray(run.X)[:, -1]


def run_point(design, point, out, start, X_prev=None, prev_id=None, progress=True):
    """Corre um ponto e grava `<id>.npz` ou `<id>.failed.json`; devolve o estado."""
    from mfsim.io import case_from_json
    from mfsim.runs import save_run
    from mfsim.simulation import simulate

    pid = point["id"]
    t0 = time.time()
    try:
        path = build_case(design, point, out, start, X_prev=X_prev, prev_id=prev_id)
        res = simulate(case_from_json(str(path)), progress=False, partial=True)
        info = dict(res.info)
        info["cpu_total"] = time.time() - t0
        info["note"] = json.dumps(dict(start=start, from_point=prev_id, chain=point["chain"], order=point["order"]))
        tmp = Path(out) / f"{pid}.partial.npz"  # gravação atómica: uma morte a meio nunca deixa um .npz truncado
        save_run(
            str(tmp),
            res.T,
            res.X,
            res.model.network,
            info=info,
            acc=info.get("accum"),
            command=f"tools/paper02_dataset.py run --out {out} (ponto {pid})",
        )
        os.replace(tmp, run_path(out, pid))
        if fail_path(out, pid).exists():
            fail_path(out, pid).unlink()
        if progress:
            st = info.get("settle") or {}
            print(
                f"{pid} ({start}): OK {st.get('reason', '?')} em t={st.get('t_end', float('nan')):.0f} s, "
                f"{info.get('n_steps')} passos, {info['cpu_total']:.0f} s",
                flush=True,
            )
        return "done"
    except Exception as e:  # noqa: BLE001  (um ponto que rebenta não trava a cadeia)
        fail_path(out, pid).write_text(
            json.dumps(dict(error=f"{type(e).__name__}: {e}", start=start, from_point=prev_id, cpu=time.time() - t0)),
            encoding="utf-8",
        )
        if progress:
            print(f"{pid} ({start}): FALHOU {type(e).__name__}: {str(e)[:120]}", flush=True)
        return "failed"


def run_chain(args):
    """Uma cadeia em sequência: retoma do último ponto concluído; para com o ficheiro STOP."""
    out, chain_id, progress = args
    design = load_design(out)
    pts = sorted((p for p in design["points"] if p["chain"] == chain_id), key=lambda p: p["order"])
    done = []
    prev_done = None
    for p in pts:
        st = point_status(out, p["id"])
        if st == "done":
            prev_done = p["id"]
            continue
        if st == "failed" and not design.get("retry_failed", False):
            continue
        if (Path(out) / "STOP").exists() or (Path(out).parent / "STOP").exists():
            if progress:
                print(f"cadeia {chain_id}: STOP encontrado, parada antes de {p['id']}", flush=True)
            break
        if prev_done is None or design.get("test"):
            st = run_point(design, p, out, "blocked", progress=progress)
        else:
            st = run_point(
                design, p, out, "warm", X_prev=_final_state(out, prev_done), prev_id=prev_done, progress=progress
            )
        if st == "done":
            prev_done = p["id"]
        done.append((p["id"], st))
    return chain_id, done


def _run_pending(out, design, jobs):
    """Corre as cadeias com pontos por fazer; devolve (feitas, falhadas)."""
    chains = sorted({p["chain"] for p in design["points"] if point_status(out, p["id"]) == "pending"})
    if not chains:
        return 0, 0
    work = [(str(out), c, True) for c in chains]
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
        master = load_rounds(root)
        cur = root / master["rounds"][-1]["folder"]
        design = load_design(cur)
        if a.retry_failed and not design.get("retry_failed"):
            design["retry_failed"] = True
            save_design(cur, design)
        pending = [p for p in design["points"] if point_status(cur, p["id"]) == "pending"]
        if pending:
            print(f"ronda {design.get('round', 0)} ({cur.name}): {len(pending)} pontos por correr", flush=True)
            n_ok, n_bad = _run_pending(cur, design, a.jobs)
            tot_ok, tot_bad = tot_ok + n_ok, tot_bad + n_bad
        if stop.exists():
            print("STOP encontrado: sem rondas novas", flush=True)
            break
        if a.rounds == 0 or (a.rounds > 0 and len(master["rounds"]) >= a.rounds):
            break
        d = create_round(root, load_rounds(root), chains=a.chains_per_round)
        print(f"ronda nova em {d}", flush=True)
    print(f"feito: {tot_ok} corridas, {tot_bad} falhas; estado em `status`", flush=True)


def cmd_run(a):
    out = Path(a.out)
    if is_rounds_root(out):
        return cmd_run_rounds(a)
    design = load_design(out)
    if a.retry_failed:
        design["retry_failed"] = True
        save_design(out, design)
    stop = out / "STOP"
    if stop.exists() and not a.keep_stop:
        stop.unlink()
    tot_ok = tot_bad = 0
    while True:
        design = load_design(out)
        pending = [p for p in design["points"] if point_status(out, p["id"]) == "pending"]
        if pending:
            chains = sorted({p["chain"] for p in pending})
            print(f"{len(pending)} pontos por correr em {len(chains)} cadeias ({a.jobs} processos)", flush=True)
            n_ok, n_bad = _run_pending(out, design, a.jobs)
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
            print(f"  {d.name}: {c['done']} feitos, {c['failed']} falhados, {c['pending']} por fazer")
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
        c = cycle_class(Tc, series["wellhead"]["P"][sel], series["riser_top"]["qL"][sel], smooth=SMOOTH)
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
        "id": "identificador do ponto",
        "chain": "cadeia (processo) a que o ponto pertence",
        "order": "posição na cadeia (0 = partida bloqueada)",
        "start": "blocked (estado bloqueado do gerador) ou warm (estado final do ponto anterior)",
        "from_point": "ponto cujo estado final arrancou esta corrida",
        "t_end": "tempo simulado alcançado [s]",
        "reason": "paragem: steady, cycle, tf (tempo máximo sem assentar), failure, none",
        "t_cap0/t_cap1": "janela de captura dos marcadores [s]",
        "class": "stable, cycling (ciclo de pressão) ou slugging (golfada severa) na janela de captura",
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


def _rows_of(out, design, tail):
    from mfsim.runs import load_run

    rows = []
    for p in design["points"]:
        if point_status(out, p["id"]) != "done":
            continue
        run = load_run(str(run_path(out, p["id"])))
        note = {}
        try:
            note = json.loads((run.info or {}).get("note") or "{}")
        except (TypeError, ValueError):
            pass
        row = dict(
            id=p["id"], chain=p["chain"], order=p["order"], start=note.get("start"), from_point=note.get("from_point")
        )
        row.update(p["x"])
        row.update(run_markers(run, tail=float(tail)))
        rows.append(row)
    return rows


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
    d.add_argument("--force", action="store_true")
    d.add_argument("--flat", action="store_true", help="conjunto numa só pasta (sem rondas datadas; o modo dos testes)")
    d.add_argument(
        "--start",
        choices=["random", "centre"],
        default="random",
        help="primeiro ponto de cada cadeia: sorteado (omissão, pedido do dono) ou o mais perto do centro",
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
        "--rounds",
        type=int,
        default=0,
        help="rondas no total: ao acabar os pontos, sortear outra ronda e recomeçar do centro (0 = só o que existe; -1 = até STOP)",
    )
    r.add_argument("--chains-per-round", type=int, default=1, help="cadeias de cada ronda nova")
    r.set_defaults(func=cmd_run)
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
