"""Figuras e tabelas do artigo 02 (gas lift: massa de dados, substitutos NN e PINN) a partir de
`runs/paper02/base2d/dataset.csv` e `runs/paper02/surrogate/` (não simula nem treina).

uso: uv run python figures/make_figures.py --out figures/generated --tables tables/generated [--only NOME] [--no-png]
     (a partir da pasta do artigo; `make figures` faz o mesmo)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_common"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figlib as fl  # noqa: E402

DATASET = fl.RUNS / "paper02" / "base2d" / "dataset.csv"
SURR = fl.RUNS / "paper02" / "surrogate"
OUTPUT_LABELS = {
    "qL_top": r"$\bar q_L$ at the top",
    "qG_top": r"$\bar q_G$ at the top",
    "q_inj": r"$\bar q_{gl}$ (gas lift)",
    "qL_span": r"span of $q_L$",
    "qG_span": r"span of $q_G$",
    "q_inj_span": r"span of $q_{gl}$",
    "P_bh": r"$\bar P_{bh}$",
    "P_inj": r"$\bar P_{inj}$",
    "P_an": r"$\bar P_{an}$",
}
CLASS_COLORS = {"stable": fl.COLORS.get("stable", "tab:green"), "cycling": fl.COLORS.get("cycling", "tab:orange"), "slugging": fl.COLORS.get("slugging", "tab:red")}


def _rows():
    if not DATASET.exists():
        return []
    return list(csv.DictReader(open(DATASET, encoding="utf-8")))


def _results():
    p = SURR / "results.json"
    return json.load(open(p)) if p.exists() else None


def _model(name):
    p = SURR / f"{name}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    m = {k: np.asarray(v, float) for k, v in d["normalization"].items()}
    m.update(d["hyperparameters"])
    m["params"] = [(np.asarray(l["W"], float), np.asarray(l["b"], float)) for l in d["layers"]]
    return m


def _predict(m, X):
    h = (np.atleast_2d(X) - m["x_mean"]) / m["x_std"]
    for W, b in m["params"][:-1]:
        h = np.tanh(h @ W + b)
    W, b = m["params"][-1]
    Y = (h @ W + b) * m["y_std"] + m["y_mean"]
    if m.get("span_log"):
        Y[:, 3:6] = np.expm1(Y[:, 3:6])
    return Y


# ------------------------------------------------------------------------------------------------ massa de dados
def dataset_map(out, png):
    """Mapa dos pontos no plano das decisões: classe, produção média e pico de líquido."""
    rows = _rows()
    fig, axs = plt.subplots(1, 3, figsize=(fl.DOUBLE, 0.36 * fl.DOUBLE), sharey=True)
    if not rows:
        for a in axs:
            fl.placeholder(a, "dataset not found (runs/paper02/base2d)")
        fl.save(fig, out, "dataset_map", png)
        return
    P = np.array([float(r["P_gas_bar"]) for r in rows])
    A = np.array([float(r["choke_opening"]) for r in rows])
    cls = np.array([r["class"] for r in rows])
    rs = np.array([r["reason"] for r in rows])
    qL = np.array([float(r["riser_top_qL_mean"]) for r in rows])
    pk = np.array([float(r["riser_top_qL_max"]) for r in rows])
    for c, col in CLASS_COLORS.items():
        m = cls == c
        axs[0].scatter(P[m], A[m], c=col, s=16, edgecolor="k", lw=0.3, label=f"{c} ({m.sum()})")
    m = rs == "failure"
    axs[0].scatter(P[m], A[m], marker="x", c="k", s=30, label=f"40-min budget exhausted ({m.sum()})")
    # legenda fora do painel (os pontos cobrem todo o plano)
    axs[0].legend(fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=2, frameon=False, handletextpad=0.3, columnspacing=0.8)
    axs[0].set_ylabel("choke opening $a$")
    sc = axs[1].scatter(P, A, c=qL, s=18, cmap="viridis", edgecolor="k", lw=0.3)
    cb = plt.colorbar(sc, ax=axs[1], pad=0.03)
    cb.set_label(r"$\bar q_L$ at the top (kg s$^{-1}$)", fontsize=7)
    sc = axs[2].scatter(P, A, c=np.log10(np.maximum(pk, 0.1)), s=18, cmap="magma", edgecolor="k", lw=0.3)
    cb = plt.colorbar(sc, ax=axs[2], pad=0.03)
    cb.set_label(r"$\log_{10}$ peak $q_L$ (kg s$^{-1}$)", fontsize=7)
    for i, a in enumerate(axs):
        a.set_xlabel("$P_{gas}$ (bar)")
        fl.panel_label(a, f"({'abc'[i]})")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.86, wspace=0.12)
    fl.save(fig, out, "dataset_map", png)


def tab_dataset(out):
    rows = _rows()
    if not rows:
        return
    cls = [r["class"] for r in rows]
    rs = [r["reason"] for r in rows]
    steps = np.array([float(r["n_steps"]) for r in rows if r["n_steps"]])
    cpu = np.array([float(r["cpu"]) for r in rows if r["cpu"]])
    t_end = np.array([float(r["t_end"]) for r in rows])
    body = [
        ["points (runs)", str(len(rows))],
        ["steady / pressure cycle / severe slugging", f"{cls.count('stable')} / {cls.count('cycling')} / {cls.count('slugging')}"],
        ["stop: steady / cycle captured / time limit / budget", f"{rs.count('steady')} / {rs.count('cycle')} / {rs.count('tf')} / {rs.count('failure')}"],
        ["simulated time per run (s): median, min--max", f"{np.median(t_end):.0f}, {t_end.min():.0f}--{t_end.max():.0f}"],
        ["accepted steps per run: median, min--max", f"{np.median(steps):.0f}, {steps.min():.0f}--{steps.max():.0f}"],
        ["integration time per run (min): median, max", f"{np.median(cpu) / 60:.1f}, {cpu.max() / 60:.1f}"],
    ]
    fl.tex_table(
        out,
        "tab_dataset",
        ["", ""],
        body,
        align="lr",
        notes=(
            "Base well of Part~I ($P_{res} = \\SI{200}{\\bar}$, $J = \\SI{5}{\\cubic\\metre\\per\\day\\per\\bar}$, GOR 50) "
            "with the gas-lift branch; inputs sampled by scrambled Sobol' sequences in rounds of 25 over $P_{gas} \\in$ "
            "[20, 160] bar and $a \\in$ [0.1, 1]; each round starts from the final state of the base run at the point "
            "closest to it and continues along the shortest path with warm starts. Budget: 40 min of integration per run."
        ),
    )


# ------------------------------------------------------------------------------------------------ substitutos
def surrogate_parity(out, png):
    """Paridade no conjunto de teste, NN (cima) e PINN (baixo), seis saídas."""
    res = _results()
    d = np.load(SURR / "data.npz", allow_pickle=True) if (SURR / "data.npz").exists() else None
    fig, axs = plt.subplots(2, 6, figsize=(fl.DOUBLE, 0.4 * fl.DOUBLE))
    if res is None or d is None:
        for a in axs.ravel():
            fl.placeholder(a, "surrogate results not found")
        fl.save(fig, out, "surrogate_parity", png)
        return
    X, Y, te = d["X"], d["Y"], d["test_idx"]
    names = res["outputs"]
    for i, k in enumerate(("nn", "pinn")):
        m = _model(k)
        Yp = _predict(m, X[te])
        for j in range(6):
            ax = axs[i, j]
            y, yp = Y[te, j], Yp[:, j]
            ax.scatter(y, yp, s=9, c=fl.COLORS.get("stratified", "tab:blue") if k == "nn" else fl.COLORS.get("annular", "tab:red"), edgecolor="k", lw=0.25)
            lo, hi = min(y.min(), yp.min()), max(y.max(), yp.max())
            ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.6, ls="--")
            ax.set_title(f"{OUTPUT_LABELS[names[j]]}\n{k.upper()}, $R^2$ = {res['full'][k]['test']['r2'][j]:.2f}", fontsize=6.5)
            ax.tick_params(labelsize=6)
            if i == 1:
                ax.set_xlabel("simulated (kg s$^{-1}$)", fontsize=7)
            if j == 0:
                ax.set_ylabel("predicted", fontsize=7)
    fig.tight_layout()
    fl.save(fig, out, "surrogate_parity", png)


def surrogate_learning(out, png):
    """Curvas de aprendizagem: RMSE de teste contra o número de pontos de treino, NN e PINN (média ± desvio)."""
    res = _results()
    fig, axs = plt.subplots(1, 3, figsize=(fl.DOUBLE, 0.3 * fl.DOUBLE))
    if res is None or not res.get("sparse"):
        for a in axs:
            fl.placeholder(a, "sparse-data study not found")
        fl.save(fig, out, "surrogate_learning", png)
        return
    sp = res["sparse"]
    sizes = sorted({r["n_train"] for r in sp}) + [res["n_train"]]
    for ax, j, name in zip(axs, (0, 1, 2), ("qL_top", "qG_top", "q_inj")):
        for k, c in (("nn", fl.COLORS.get("stratified", "tab:blue")), ("pinn", fl.COLORS.get("annular", "tab:red"))):
            mean, sd = [], []
            for n in sizes:
                v = [r[k]["test"]["rmse"][j] for r in sp if r["n_train"] == n] if n != res["n_train"] else [res["full"][k]["test"]["rmse"][j]]
                mean.append(np.mean(v))
                sd.append(np.std(v))
            ax.errorbar(sizes, mean, yerr=sd, marker="o", ms=3, color=c, label=k.upper(), capsize=2, lw=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("training points")
        ax.set_ylabel(f"test RMSE, {OUTPUT_LABELS[name]} (kg s$^{{-1}}$)")
        ax.legend(fontsize=6.5)
    for i, a in enumerate(axs):
        fl.panel_label(a, f"({'abc'[i]})")
    fig.tight_layout()
    fl.save(fig, out, "surrogate_learning", png)


def surrogate_maps(out, png):
    """Mapas do PINN no plano das decisões (produção média de líquido, span de líquido, gás de lift) com os dados."""
    d = np.load(SURR / "data.npz", allow_pickle=True) if (SURR / "data.npz").exists() else None
    mp = np.load(SURR / "pinn_map.npz") if (SURR / "pinn_map.npz").exists() else None
    fig, axs = plt.subplots(1, 3, figsize=(fl.DOUBLE, 0.34 * fl.DOUBLE))
    if d is None or mp is None:
        for a in axs:
            fl.placeholder(a, "PINN maps not found")
        fl.save(fig, out, "surrogate_maps", png)
        return
    X, Y = d["X"], d["Y"]
    # curvas de nível (pedido do dono), com os pontos simulados só como posições
    specs = (
        (axs[0], 0, r"$\bar q_L$ at the top (kg s$^{-1}$)", np.arange(3.0, 5.51, 0.25), "%.2f"),
        (axs[1], 3, r"span of $q_L$ (kg s$^{-1}$)", np.array([1.0, 5.0, 10.0, 20.0, 30.0, 40.0]), "%.0f"),
        (axs[2], 2, r"$\bar q_{gl}$ (kg s$^{-1}$)", np.arange(0.1, 1.21, 0.1), "%.1f"),
    )
    for ax, j, lab, levels, fmt in specs:
        Z = np.maximum(mp["Y"][:, :, j], 0.0) if j == 3 else mp["Y"][:, :, j]
        cs = ax.contour(mp["P"], mp["a"], Z, levels=levels, colors=fl.INK, linewidths=0.7)
        ax.clabel(cs, fmt=fmt, fontsize=6, inline=True, inline_spacing=2)
        ax.scatter(X[:, 1], X[:, 0], s=5, c="0.55", edgecolor="none", zorder=0)
        ax.set_xlabel("$P_{gas}$ (bar)")
        ax.set_title(lab, fontsize=8)
    axs[0].set_ylabel("choke opening $a$")
    for i, a in enumerate(axs):
        fl.panel_label(a, f"({'abc'[i]})")
    fig.tight_layout()
    fl.save(fig, out, "surrogate_maps", png)


def tab_surrogate(out):
    res = _results()
    if res is None:
        return
    names = res["outputs"]
    rows = []
    for j, n in enumerate(names):
        rows.append(
            [
                OUTPUT_LABELS[n],
                fl.fmt(res["full"]["nn"]["test"]["r2"][j], "{:.3f}"),
                fl.fmt(res["full"]["nn"]["test"]["rmse"][j], "{:.3f}"),
                fl.fmt(res["full"]["pinn"]["test"]["r2"][j], "{:.3f}"),
                fl.fmt(res["full"]["pinn"]["test"]["rmse"][j], "{:.3f}"),
            ]
        )
    rr = ["residuals $R_1$--$R_4$ (RMS, std units)", "--", ", ".join(f"{v:.3f}" for v in res["full"]["nn"]["residual_rms"]), "--", ", ".join(f"{v:.3f}" for v in res["full"]["pinn"]["residual_rms"])]
    rows.append(rr)
    fl.tex_table(
        out,
        "tab_surrogate",
        ["output", "NN $R^2$", "NN RMSE", "PINN $R^2$", "PINN RMSE"],
        rows,
        align="lrrrr",
        notes=(
            f"Test set of {res['n_test']} runs ({res['n_train']} for training); RMSE in kg/s for the flow rates and bar for the "
            "pressures. Residuals: root-mean-square of the four scaled physics residuals on 400 held-out collocation points. "
            "Both networks have two hidden layers of 32 hyperbolic-tangent units; the PINN adds the physics term with "
            "$\\lambda = 1$ on 200 Latin-hypercube collocation points."
        ),
    )


def tab_sparse(out):
    res = _results()
    if res is None or not res.get("sparse"):
        return
    sp = res["sparse"]
    rows = []
    for n in sorted({r["n_train"] for r in sp}):
        sel = [r for r in sp if r["n_train"] == n]
        row = [str(n)]
        for j in (0, 1, 2):
            for k in ("nn", "pinn"):
                v = [r[k]["test"]["rmse"][j] for r in sel]
                row.append(f"{np.mean(v):.3f} $\\pm$ {np.std(v):.3f}")
        rows.append(row)
    row = [str(res["n_train"])]
    for j in (0, 1, 2):
        for k in ("nn", "pinn"):
            row.append(f"{res['full'][k]['test']['rmse'][j]:.3f}")
    rows.append(row)
    fl.tex_table(
        out,
        "tab_sparse",
        ["training points", "$\\bar q_L$ NN", "$\\bar q_L$ PINN", "$\\bar q_G$ NN", "$\\bar q_G$ PINN", "$\\bar q_{gl}$ NN", "$\\bar q_{gl}$ PINN"],
        rows,
        align="rrrrrrr",
        notes="Test RMSE (kg/s), mean $\\pm$ standard deviation over three random subsets of the training set; last row: the full training set.",
    )


def system(out, png):
    """Esquema do sistema com o ramo de gas lift: a figura da geometria do artigo 01 (o mesmo guião, lê os casos)."""
    import importlib.util

    p01 = Path(__file__).resolve().parents[2] / "01-severe-slugging-two-fluid" / "figures" / "make_figures.py"
    spec = importlib.util.spec_from_file_location("paper01_figures", p01)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(p01.parent))
    spec.loader.exec_module(mod)
    mod.geometry(out, png)
    for ext in ("pdf", "png"):
        src = Path(out) / f"geometry.{ext}"
        if src.exists():
            src.replace(Path(out) / f"system.{ext}")


FIGURES = {
    "system": system,
    "dataset_map": dataset_map,
    "surrogate_parity": surrogate_parity,
    "surrogate_learning": surrogate_learning,
    "surrogate_maps": surrogate_maps,
}
TABLES = {"tab_dataset": tab_dataset, "tab_surrogate": tab_surrogate, "tab_sparse": tab_sparse}

if __name__ == "__main__":
    fl.run_cli(FIGURES, __doc__.splitlines()[0], tables=TABLES)
