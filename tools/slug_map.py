"""Mapa de intermitência severa (issue 043): abertura do choke × pressão do reservatório no caso
do poço (`cases/severe_slug/well_choke.json`: a referência com IPR linear e choke homogéneo).

Tudo passa pelo caminho único (casos JSON corridos por `mfsim batch`):

    uv run python tools/slug_map.py build --out runs/maps/well           # casos e lotes
    uv run mfsim batch runs/maps/well/steady/batch.json --jobs 12        # partida do estacionário
    uv run mfsim batch runs/maps/well/blocked/batch.json --jobs 12       # partida bloqueada
    uv run python tools/slug_map.py plot --out runs/maps/well            # resumo e figura

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


def _base_params():
    d = json.load(open(BASE))
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
    """Estados estacionários de toda a grelha {(i, j): X ou None}: coluna da semente em P,
    depois cada linha em abertura, para os dois lados."""
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
            row.update(run_cycle_class(run))
            row.update(
                success=bool(run.info.get("success", True)),
                settle=(run.info.get("settle") or {}).get("reason"),
                initial=(run.info.get("initial") or {}).get("converged"),
                n_steps=run.info.get("n_steps"),
                cpu=run.info.get("cpu_total"),
            )
            rows.append(row)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)
    return rows


# ----------------------------------------------------------------------------- figura
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
COLORS = {"stable": "#2a78d6", "slugging": "#eb6834", "cycling": "#1baf7a", "fail": "#e1e0d9"}
LABELS = {
    "stable": "Estável: escoamento permanente",
    "slugging": "Golfada severa: pico na saída ≥ 2× a média",
    "cycling": "Ciclo de pressão, saída suave",
    "fail": "Corrida falhou",
}


def _edges(v, log=False):
    v = np.log(v) if log else np.asarray(v, float)
    mid = 0.5 * (v[1:] + v[:-1])
    e = np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])
    return np.exp(e) if log else e


def jansen_dp_crit(alpha=0.74, alpha_p=0.6, L=320.0, H=20.0, P_S=4e5, Phi=1.0, rhoL=900.0, g=9.81):
    """Queda no choke (bar, relação monofásica do líquido) acima da qual o critério de Jansen,
    Shoham e Taitel (1996) prevê estabilidade (eq. 5 com Phi = 1, eq. 10 com a fração de líquido
    do riser em permanente). Insensível a alpha e alpha' nesta geometria (0,53–0,76 bar)."""
    aL = alpha * L / alpha_p
    return (Phi * rhoL * g * (aL - H) - P_S) / (1.0 + 2.0 * aL / H) / 1e5


def plot(out, rows=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    rows = rows if rows is not None else json.load(open(os.path.join(out, "summary.json")))
    plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 10})
    abs_ = sorted({r["ab"] for r in rows})
    Ps = sorted({r["P_res"] for r in rows})
    xe, ye = _edges(abs_, log=True), _edges(Ps, log=True)
    titles = {
        "steady": "Partida do estado estacionário (perturbação 0,1 %)",
        "blocked": "Partida bloqueada (linha cheia, riser bloqueado)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.2), sharey=True, facecolor=SURF)
    dpc = 0.5 * (jansen_dp_crit(Phi=0.87) + jansen_dp_crit())
    for ax, start in zip(axes, ("steady", "blocked")):
        ax.set_facecolor(SURF)
        G = {(r["P_res"], r["ab"]): r for r in rows if r["start"] == start}
        for i, P in enumerate(Ps):
            best = None
            for j, ab in enumerate(abs_):
                r = G.get((P, ab))
                if r is None:
                    continue
                cls = r["cls"] if r["success"] else "fail"
                ax.add_patch(
                    Rectangle((xe[j], ye[i]), xe[j + 1] - xe[j], ye[i + 1] - ye[i], fc=COLORS[cls], ec=SURF, lw=1.5)
                )
                xc, yc = np.sqrt(xe[j] * xe[j + 1]), np.sqrt(ye[i] * ye[i + 1])
                if cls == "stable":
                    txt = f"{r['P_mean']:.1f}"
                    if best is None or r["q_mean"] > best[1]:
                        best = (ab, r["q_mean"], j)
                else:
                    per = r["period"]
                    txt = (
                        f"{r['P_max']:.1f}\n{per / 60:.0f} min"
                        if np.isfinite(per) and per >= 120
                        else (f"{r['P_max']:.1f}\n{per:.0f} s" if np.isfinite(per) else f"{r['P_max']:.1f}")
                    )
                ax.text(
                    xc, yc, txt.replace(".", ","), ha="center", va="center", fontsize=6.6, color=INK, linespacing=1.05
                )
            if best is not None:
                j = best[2]
                ax.add_patch(
                    Rectangle((xe[j], ye[i]), xe[j + 1] - xe[j], ye[i + 1] - ye[i], fc="none", ec=INK, lw=2.0, zorder=5)
                )
        # critério de Jansen et al. (1996): abertura em que a queda monofásica no choke iguala o limiar
        xs, ys = [], []
        for P in Ps:
            pts = sorted([(ab, G[(P, ab)]["q_mean"]) for ab in abs_ if (P, ab) in G], key=lambda t: t[0])
            ab = np.array([p[0] for p in pts])
            q = np.array([p[1] for p in pts])
            dpl = q**2 / 900.0 / (0.0075 * 15.0 * ab) ** 2
            if dpl.max() > dpc > dpl.min():
                xs.append(float(np.exp(np.interp(-np.log(dpc), -np.log(dpl), np.log(ab)))))
                ys.append(P)
        ax.plot(xs, ys, color=INK, lw=1.6, ls=(0, (5, 3)), zorder=6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = [t for t in (0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0) if xe[0] <= t <= xe[-1]]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}".replace(".", ",") for t in ticks])
        ax.set_yticks(Ps)
        ax.set_yticklabels([f"{P:g}".replace(".", ",") for P in Ps])
        ax.minorticks_off()
        ax.set_xlim(xe[0], xe[-1])  # depois das marcas: set_xticks alarga os limites
        ax.set_ylim(ye[0], ye[-1])
        ax.set_xlabel("Abertura do choke (fração, escala log)")
        ax.set_title(titles[start], loc="left", fontsize=11, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Pressão do reservatório (bar)")
    handles = [Patch(fc=COLORS[k], label=LABELS[k]) for k in ("stable", "cycling", "slugging")]
    if any(not r["success"] for r in rows):
        handles.append(Patch(fc=COLORS["fail"], label=LABELS["fail"]))
    handles += [
        Patch(fc="none", ec=INK, lw=2.0, label="Abertura estável mais produtiva"),
        Line2D(
            [],
            [],
            color=INK,
            lw=1.6,
            ls=(0, (5, 3)),
            label="Critério de Jansen et al. 1996 (queda no choke " + f"{dpc:.2f}".replace(".", ",") + " bar)",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "Mapa de intermitência severa: abertura do choke × pressão do reservatório",
        x=0.05,
        ha="left",
        fontsize=13,
        color=INK,
    )
    fig.text(
        0.05,
        0.915,
        "Em cada célula: estável, a pressão de entrada da linha (bar); com ciclo, a pressão máxima (bar) e o período. "
        "Caso de referência como poço (IPR 0,056 kg/s/bar), choke homogéneo CV 15, separador a 4 bar; "
        "paragem adaptativa até 8000 s. Estável: amplitude da pressão < 0,1 bar ou < 1 % da média, ou sem ciclo até aos 8000 s.",
        fontsize=8.6,
        color=INK2,
        wrap=True,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.87, bottom=0.17, wspace=0.04)
    path = os.path.join(out, "slug_map.png")
    fig.savefig(path, dpi=150, facecolor=SURF)
    print(f"figura: {path}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["build", "plot"])
    ap.add_argument("--out", default="runs/maps/well")
    ap.add_argument("--openings", help="aberturas separadas por vírgulas (a semente 0.13 tem de estar)")
    ap.add_argument("--pressures", help="pressões do reservatório separadas por vírgulas (a semente 15 tem de estar)")
    a = ap.parse_args(argv)
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    if a.command == "build":
        ops = [float(x) for x in a.openings.split(",")] if a.openings else OPENINGS
        prs = [float(x) for x in a.pressures.split(",")] if a.pressures else PRESSURES
        build(out, ops, prs)
    else:
        plot(out, summarize(out))


if __name__ == "__main__":
    main()
