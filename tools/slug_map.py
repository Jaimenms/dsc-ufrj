"""Mapa de intermitência severa (issue 043): abertura do choke × pressão do reservatório no caso
do poço (`cases/severe_slug/well_choke.json`: a referência com IPR linear e choke homogéneo).

Tudo passa pelo caminho único (casos JSON corridos por `mfsim batch`):

    uv run python tools/slug_map.py build --out runs/maps/well           # casos e lotes
    uv run mfsim batch runs/maps/well/steady/batch.json --jobs 12        # partida do estacionário
    uv run mfsim batch runs/maps/well/blocked/batch.json --jobs 12       # partida bloqueada
    uv run python tools/slug_map.py plot --out runs/maps/well            # resumo e figura
    uv run python tools/slug_map.py export --out runs/maps/well          # dados do notebook (notebooks/data)

Outro caso base (por exemplo o do artigo 01, com padrão de escoamento dinâmico) com `--base` e outro nome
para os dados exportados com `--name`:

    uv run python tools/slug_map.py build --base cases/severe_slug/well_choke_auto.json --out runs/maps/well_auto
    uv run python tools/slug_map.py export --base cases/severe_slug/well_choke_auto.json --out runs/maps/well_auto \
        --name severe_slug_map_well_auto

`build` faz a continuação do estado estacionário pela grelha (Newton de ponto vizinho em ponto
vizinho, com bissecção do passo quando falha; semente: uma corrida que assenta) e grava-o como
condição inicial de cada caso (`initial_state.source = "steady"`, perturbação 1e-3; issue 040).
Escreve também a partida bloqueada. Os casos usam a paragem adaptativa (`time.settle`, issue
041) com 8000 s de máximo: o que não cicla até lá conta como estável (`mfsim.analysis.cycle_class`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASE = os.path.join(ROOT, "cases", "severe_slug", "well_choke.json")
OPENINGS = [0.03, 0.045, 0.06, 0.08, 0.1, 0.13, 0.17, 0.22, 0.28, 0.36, 0.46, 0.6, 0.8, 1.0]
PRESSURES = [8.0, 10.0, 12.0, 15.0, 19.0, 24.0, 30.0, 38.0, 48.0]
SEED = (0.13, 15.0)  # ponto estável do mapa (assenta depressa a partir da partida bloqueada)
SETTLE = {"window": 2000.0, "t_min": 2000.0, "amp_tol": 0.1, "rel_tol": 0.1}
TF, DT_OUT, PERTURB = 8000.0, 5.0, 1e-3
# passo máximo do integrador nas corridas do mapa (issue 047): sem limite, perto de um equilíbrio o controlo de
# erro deixa o passo crescer para centenas de segundos e o amortecimento do BDF com |lambda h| grande mantém a
# trajetória num equilíbrio instável (a partida do estacionário perturbado não sai dele)
MAX_STEP = 1.0
# média móvel (s) da pressão de entrada na classificação (`--smooth`): 0 no mapa 043; o artigo 01 usa 20 s,
# acima das oscilações de alguns segundos da coluna do riser contra o gás do bolsão e abaixo dos ciclos
SMOOTH = 0.0


def _base_params():
    """IPR do líquido e razão gás–líquido do reservatório do caso base (com a herança `$extends` resolvida)."""
    from mfsim.io.spec import load_json

    d = load_json(BASE)
    res = next(e for e in d["network"]["edges"] if e["id"] == "reservoir_1")
    return res["IPL"], res["IPG"] / res["IPL"]


def overrides(ab, P):
    """Overrides de um ponto da grelha; as vazões da junção e da receita são só a estimativa."""
    ipl, glr = _base_params()
    q0 = max(0.02, ipl * (P - 6.0))
    return {
        "network.edges.choke_2.opening": float(ab),
        "network.edges.reservoir_1.P_bar": float(P),
        "network.vertices.Jr1.spec.qL": q0,
        "network.vertices.Jr1.spec.qG": q0 * glr,
        "network.edges.1.ic.qL": q0,
        "network.edges.1.ic.qG": q0 * glr,
    }


def point_case(ab, P, start):
    from mfsim.io import case_from_json

    net = case_from_json(BASE, overrides=overrides(ab, P))
    net.set_time(tf=TF, dt_out=DT_OUT, settle=tuple(sorted(SETTLE.items())))
    net.set_integration(max_step=MAX_STEP)
    net.meta = dict(net.meta or {}, map={"ab": float(ab), "P_res": float(P), "start": start})
    return net


def name(ab, P):
    return f"ab{ab:.3f}_P{P:04.1f}"


# ----------------------------------------------------------------------------- continuação
def _newton(ab, P, X0):
    from mfsim.simulation import build_model, state_bounds
    from mfsim.steady import newton_steady

    model = build_model(point_case(ab, P, "steady"))
    X, info = newton_steady(model, X0, tol=1e-9, verbose=False, bounds=state_bounds(model), jax=True, max_iter=60)
    return X if info["converged"] else None


def _path(X, a0, P0, a1, P1, depth=0):
    X1 = _newton(a1, P1, X)
    if X1 is not None or depth >= 5:
        return X1
    am, Pm = float(np.sqrt(a0 * a1)), 0.5 * (P0 + P1)
    Xm = _path(X, a0, P0, am, Pm, depth + 1)
    return None if Xm is None else _path(Xm, am, Pm, a1, P1, depth + 1)


def seed_state():
    """Estado estacionário na semente: corrida da partida bloqueada até assentar e Newton."""
    from mfsim.simulation import simulate

    net = point_case(*SEED, "blocked")
    res = simulate(net, progress=False)
    if res.info.get("settle", {}).get("reason") != "steady":
        raise RuntimeError(f"a semente {SEED} não assentou: {res.info.get('settle', {}).get('reason')}")
    X = _newton(*SEED, res.X[:, -1])
    if X is None:
        raise RuntimeError("Newton falhou na semente")
    return X


def continuation(openings, pressures, log=print):
    """Estados estacionários de toda a grelha {(i, j): X ou None}: coluna da semente em P, depois cada
    linha em abertura, para os dois lados, e por fim passagens de preenchimento: cada ponto que falhou
    tenta a partir de qualquer vizinho convergido (abertura ou pressão ao lado), até não haver progresso.
    Com o padrão dinâmico os fechamentos têm quinas (tabelas de equilíbrio, máximos suaves) e um passo de
    continuação pode estagnar numa direção e passar noutra."""
    X = _continuation_sweeps(openings, pressures, log)
    while True:
        new = 0
        for i in range(len(pressures)):
            for j in range(len(openings)):
                if X.get((i, j)) is not None:
                    continue
                for di, dj in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    Xn = X.get((i + di, j + dj))
                    if Xn is None:
                        continue
                    Xs = _path(Xn, openings[j + dj], pressures[i + di], openings[j], pressures[i])
                    if Xs is not None:
                        X[(i, j)] = Xs
                        new += 1
                        break
        log(f"  preenchimento: +{new} ({sum(x is not None for x in X.values())}/{len(openings) * len(pressures)})")
        if not new:
            return X


def _continuation_sweeps(openings, pressures, log=print):
    """Coluna da semente em P, depois cada linha em abertura, para os dois lados."""
    j0, i0 = openings.index(SEED[0]), pressures.index(SEED[1])
    X = {(i0, j0): seed_state()}
    for rng in (range(i0 + 1, len(pressures)), range(i0 - 1, -1, -1)):
        prev = i0
        for i in rng:
            X[(i, j0)] = (
                _path(X[(prev, j0)], openings[j0], pressures[prev], openings[j0], pressures[i])
                if X.get((prev, j0)) is not None
                else None
            )
            log(f"  coluna ab={openings[j0]}: P={pressures[i]} {'ok' if X[(i, j0)] is not None else 'FALHOU'}")
            prev = i
    for i in range(len(pressures)):
        for rng in (range(j0 + 1, len(openings)), range(j0 - 1, -1, -1)):
            prev = j0
            for j in rng:
                Xp = X.get((i, prev))
                X[(i, j)] = None if Xp is None else _path(Xp, openings[prev], pressures[i], openings[j], pressures[i])
                prev = j
        log(f"  linha P={pressures[i]}: {sum(X.get((i, j)) is not None for j in range(len(openings)))}/{len(openings)}")
    return X


# ----------------------------------------------------------------------------- construção
def build(out, openings=OPENINGS, pressures=PRESSURES):
    from mfsim.io import case_to_json
    from mfsim.simulation import build_model

    states = continuation(openings, pressures)
    for start in ("steady", "blocked"):
        cdir = os.path.join(out, start, "cases")
        os.makedirs(cdir, exist_ok=True)
        rel = os.path.relpath(BASE, cdir)
        runs = []
        for i, P in enumerate(pressures):
            for j, ab in enumerate(openings):
                net = point_case(ab, P, start)
                if start == "steady":
                    X = states.get((i, j))
                    if X is not None:
                        model = build_model(net)
                        model.state_to_case_ic(X, model.t0, exact=True)  # nós do esquema: ida e volta exata
                        net.initial = "steady"
                        net.initial_options = {"from": "edges", "perturb": PERTURB, "on_failure": "start"}
                    else:
                        net.meta["map"]["start"] = "blocked (continuação falhou)"
                path = os.path.join(cdir, f"{name(ab, P)}.json")
                case_to_json(net, path, extends=rel)
                runs.append({"name": name(ab, P), "case": os.path.relpath(path, ROOT)})
        batch = {"out_dir": os.path.relpath(os.path.join(out, start, "runs"), ROOT), "runs": runs}
        with open(os.path.join(out, start, "batch.json"), "w") as f:
            json.dump(batch, f, indent=1)
        print(f"{start}: {len(runs)} casos em {cdir}")
    n_ok = sum(x is not None for x in states.values())
    print(f"continuação: {n_ok}/{len(openings) * len(pressures)} estacionários")


# ----------------------------------------------------------------------------- resumo
def summarize(out):
    """Linhas do mapa (classe e assinatura de cada corrida, `run_cycle_class`), com o estacionário de
    partida (pressão de entrada e vazão de saída no instante inicial das corridas da partida do
    estacionário, sem a perturbação) em `P_ss`/`q_ss`; grava `summary.json` em `out`."""
    from mfsim.analysis import run_cycle_class
    from mfsim.runs import load_run

    rows = []
    for start in ("steady", "blocked"):
        batch = json.load(open(os.path.join(out, start, "batch.json")))
        rdir = os.path.join(ROOT, batch["out_dir"])
        for r in batch["runs"]:
            path = os.path.join(rdir, f"{r['name']}.npz")
            if not os.path.exists(path):
                continue
            run = load_run(path)
            m = (run.case_json.get("meta") or {}).get("map") or {}
            row = dict(start=start, name=r["name"], ab=m.get("ab"), P_res=m.get("P_res"), start_used=m.get("start"))
            row.update(run_cycle_class(run, smooth=SMOOTH))
            init = run.info.get("initial") or {}
            p_out = run.pipe(0)["out"]
            row.update(
                success=bool(run.info.get("success", True)),
                settle=(run.info.get("settle") or {}).get("reason"),
                initial=init.get("converged"),
                n_steps=run.info.get("n_steps"),
                cpu=run.info.get("cpu_total"),
                P_ss=(
                    float(p_out["P_bar"][1][0] / (1.0 + init.get("perturb", 0.0)))
                    if init.get("converged")
                    else float("nan")
                ),
                q_ss=float(p_out["qL"][-1][0]) if init.get("converged") else float("nan"),
            )
            rows.append(row)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)
    return rows


# ----------------------------------------------------------------------------- figura
TITLES = {
    "steady": "Partida do estado estacionário (perturbação 0,1 %)",
    "blocked": "Partida bloqueada (linha cheia, riser bloqueado)",
}


def plot_map(rows, axes=None):
    """Os dois painéis do mapa (partida do estacionário e bloqueada) com o critério de Jansen; devolve a figura."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from mfsim.analysis import jansen_choke_dp_crit, jansen_choke_opening
    from mfsim.plotting import CYCLE_COLORS, CYCLE_LABELS, cycle_map

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8.2), sharey=True, facecolor="#fcfcfb")
    else:
        fig = axes[0].figure
    dpc = 0.5 * (jansen_choke_dp_crit(Phi=0.87) + jansen_choke_dp_crit())
    for ax, start in zip(axes, ("steady", "blocked")):
        sub = [r for r in rows if r["start"] == start]
        xe, ye = cycle_map(ax, sub)
        xs, ys = [], []
        for P in sorted({r["P_res"] for r in sub}):
            line = sorted((r["ab"], r["q_mean"]) for r in sub if r["P_res"] == P)
            a_c = jansen_choke_opening([t[0] for t in line], [t[1] for t in line], dpc, CV=15.0)
            if a_c is not None:
                xs.append(a_c)
                ys.append(P)
        ax.plot(xs, ys, color="#0b0b0b", lw=1.6, ls=(0, (5, 3)), zorder=6)
        ticks = [t for t in (0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0) if xe[0] <= t <= xe[-1]]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}".replace(".", ",") for t in ticks])
        ax.set_xlim(xe[0], xe[-1])
        ax.set_xlabel("Abertura do choke (fração, escala log)")
        ax.set_title(TITLES[start], loc="left", fontsize=11)
    axes[0].set_ylabel("Pressão do reservatório (bar)")
    handles = [Patch(fc=CYCLE_COLORS[k], label=CYCLE_LABELS[k]) for k in ("stable", "cycling", "slugging")]
    if any(not r["success"] for r in rows):
        handles.append(Patch(fc=CYCLE_COLORS["fail"], label=CYCLE_LABELS["fail"]))
    handles += [
        Patch(fc="none", ec="#0b0b0b", lw=1.8, label="Abertura estável mais produtiva"),
        Line2D(
            [],
            [],
            color="#0b0b0b",
            lw=1.6,
            ls=(0, (5, 3)),
            label="Critério de Jansen et al. 1996 (queda no choke " + f"{dpc:.2f}".replace(".", ",") + " bar)",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
    return fig


def plot(out, rows=None):
    rows = rows if rows is not None else json.load(open(os.path.join(out, "summary.json")))
    fig = plot_map(rows)
    fig.suptitle(
        "Mapa de intermitência severa: abertura do choke × pressão do reservatório", x=0.05, ha="left", fontsize=13
    )
    fig.text(
        0.05,
        0.915,
        "Em cada célula: estável, a pressão de entrada da linha (bar); com ciclo, a pressão máxima (bar) e o período. "
        "Caso de referência como poço (IPR 0,056 kg/s/bar), choke homogéneo CV 15, separador a 4 bar; "
        "paragem adaptativa até 8000 s. Estável: amplitude da pressão < 0,1 bar ou < 1 % da média, ou sem ciclo até aos 8000 s.",
        fontsize=8.6,
        color="#52514e",
        wrap=True,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.87, bottom=0.17, wspace=0.04)
    path = os.path.join(out, "slug_map.png")
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    print(f"figura: {path}")
    return path


# ----------------------------------------------------------------------------- dados do notebook
DATA = os.path.join(ROOT, "notebooks", "data")
DATA_NAME = "severe_slug_map_well"
# corridas guardadas por inteiro para os perfis: abertura crescente a 15 bar (estável, ciclo,
# golfada) e a célula de histerese a 24 bar nas duas partidas
SELECTED = [
    ("steady", 0.13, 15.0),
    ("steady", 0.28, 15.0),
    ("steady", 0.6, 15.0),
    ("steady", 0.6, 24.0),
    ("blocked", 0.6, 24.0),
]
SUMMARY_KEYS = (
    "ab",
    "P_res",
    "P_min",
    "P_mean",
    "P_max",
    "amp",
    "period",
    "q_mean",
    "q_peak",
    "surge",
    "t_end",
    "n_steps",
    "cpu",
    "P_ss",
    "q_ss",
)


def export(out):
    """Grava em `notebooks/data` o resumo do mapa (`severe_slug_map_well.npz`: uma linha por corrida,
    colunas de `SUMMARY_KEYS` e as de texto `start`, `name`, `cls`, `settle`, mais `meta` em JSON) e as
    corridas de `SELECTED` (`severe_slug_map_well_<partida>_<ponto>.npz`), para o notebook de issues
    as ler sem as 252 corridas; registar depois com `tools/manifest.py --write`."""
    import shutil
    import subprocess

    rows = summarize(out)
    arrays = {k: np.array([float(r.get(k) if r.get(k) is not None else np.nan) for r in rows]) for k in SUMMARY_KEYS}
    for k in ("start", "name", "cls", "settle"):
        arrays[k] = np.array([str(r.get(k)) for r in rows])
    arrays["initial"] = np.array([bool(r.get("initial")) for r in rows])
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    meta = dict(
        source="tools/slug_map.py (build, mfsim batch, export)",
        base_case=os.path.relpath(BASE, ROOT),
        settle=SETTLE,
        tf=TF,
        dt_out=DT_OUT,
        perturb=PERTURB,
        seed=SEED,
        commit=commit,
        selected=[f"{DATA_NAME}_{s}_{name(a, P)}" for s, a, P in SELECTED],
    )
    arrays["meta"] = np.array(json.dumps(meta))
    os.makedirs(DATA, exist_ok=True)
    np.savez(os.path.join(DATA, f"{DATA_NAME}.npz"), **arrays)
    print(f"resumo: {DATA}/{DATA_NAME}.npz ({len(rows)} corridas)")
    for s, a, P in SELECTED:
        batch = json.load(open(os.path.join(out, s, "batch.json")))
        src = os.path.join(ROOT, batch["out_dir"], f"{name(a, P)}.npz")
        dst = os.path.join(DATA, f"{DATA_NAME}_{s}_{name(a, P)}.npz")
        shutil.copy(src, dst)
        print(f"corrida: {dst}")


def main(argv=None):
    global BASE, DATA_NAME, SMOOTH
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["build", "plot", "export"])
    ap.add_argument("--out", default="runs/maps/well")
    ap.add_argument("--openings", help="aberturas separadas por vírgulas (a semente 0.13 tem de estar)")
    ap.add_argument("--pressures", help="pressões do reservatório separadas por vírgulas (a semente 15 tem de estar)")
    ap.add_argument("--base", help="caso base (JSON com o reservatório reservoir_1 e o choke choke_2)")
    ap.add_argument("--name", help=f"nome dos dados exportados em notebooks/data (padrão {DATA_NAME})")
    ap.add_argument(
        "--smooth",
        type=float,
        default=SMOOTH,
        help="média móvel (s) da pressão de entrada antes da classe e do período (`cycle_class`; 0 = registo cru)",
    )
    a = ap.parse_args(argv)
    SMOOTH = a.smooth
    if a.base:
        BASE = a.base if os.path.isabs(a.base) else os.path.join(ROOT, a.base)
    if a.name:
        DATA_NAME = a.name
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    if a.command == "build":
        ops = [float(x) for x in a.openings.split(",")] if a.openings else OPENINGS
        prs = [float(x) for x in a.pressures.split(",")] if a.pressures else PRESSURES
        build(out, ops, prs)
    elif a.command == "plot":
        plot(out, summarize(out))
    else:
        export(out)


if __name__ == "__main__":
    main()
