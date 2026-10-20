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

DATASETS = (  # 20–160 bar; 160–260 bar; 160–260 bar com abertura 0,5–1,0 (pedido do dono: mais pontos onde está o ótimo)
    fl.RUNS / "paper02" / "base2d" / "dataset.csv",
    fl.RUNS / "paper02" / "base2d_hi" / "dataset.csv",
    fl.RUNS / "paper02" / "base2d_hi_open" / "dataset.csv",
    fl.RUNS / "paper02" / "extra" / "dataset.csv",  # corridas de verificação que assentaram (§6), mesmas condições
)
SURR = fl.RUNS / "paper02" / "surrogate"
OUTPUT_LABELS = {
    "qL_top": r"$\bar q_L$ at the top",
    "qG_top": r"$\bar q_G$ at the top",
    "q_inj": r"$\bar q_{gl}$ (gas lift)",
    "qL_span": r"span of $q_L$",
    "qG_span": r"span of $q_G$",
    "q_inj_span": r"span of $q_{gl}$",
    "qL_max": r"max. $q_L$ at the top",
    "q_inj_max": r"max. $q_{gl}$",
    "P_bh": r"$\bar P_{bh}$",
    "P_inj": r"$\bar P_{inj}$",
    "P_an": r"$\bar P_{an}$",
}
CLASS_COLORS = {
    "stable": fl.COLORS.get("stable", "tab:green"),
    "fast": fl.COLORS.get("fast", "tab:purple"),  # 062: ciclo rápido do topo do riser (período < 300 s)
    "cycling": fl.COLORS.get("cycling", "tab:orange"),
    "slugging": fl.COLORS.get("slugging", "tab:red"),
}


def _rows():
    rows = []
    for p in DATASETS:
        if p.exists():
            rows += list(csv.DictReader(open(p, encoding="utf-8")))
    return rows


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
    m["outputs"] = list(d["outputs"])
    return m


def _predict(m, X):
    """Previsão em unidades físicas: spans e máximos (colunas cujo nome tem "span" ou "max") saem de log(1 + x)."""
    h = (np.atleast_2d(X) - m["x_mean"]) / m["x_std"]
    for W, b in m["params"][:-1]:
        h = np.tanh(h @ W + b)
    W, b = m["params"][-1]
    Y = (h @ W + b) * m["y_std"] + m["y_mean"]
    if m.get("span_log"):
        for j, n in enumerate(m["outputs"]):
            if "span" in n or "max" in n:
                Y[:, j] = np.expm1(Y[:, j])
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
    axs[0].scatter(P[m], A[m], marker="x", c="k", s=30, label=f"budget exhausted ({m.sum()})")
    # legenda fora do painel (os pontos cobrem todo o plano)
    axs[0].legend(
        fontsize=6,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.04),
        ncol=2,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.8,
    )
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
        [
            "steady / fast cycle / pressure cycle / severe slugging",
            f"{cls.count('stable')} / {cls.count('fast')} / {cls.count('cycling')} / {cls.count('slugging')}",
        ],
        [
            "stop: steady / cycle captured / time limit / budget",
            f"{rs.count('steady')} / {rs.count('cycle')} / {rs.count('tf')} / {rs.count('failure')}",
        ],
        [
            "simulated time per run (s): median, min--max",
            f"{np.median(t_end):.0f}, {t_end.min():.0f}--{t_end.max():.0f}",
        ],
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
            "with the gas-lift branch; inputs sampled by scrambled Sobol' sequences in rounds of 25 over $a \\in$ [0.1, 1] and "
            "$P_{gas} \\in$ [20, 160] bar (first campaign), [160, 260] bar (second) and [160, 260] bar with $a \\in$ [0.5, 1] (third, "
            "densifying the region of the optimum); every run starts from the final state of the base run (the first rounds "
            "chained warm starts along the shortest path, Section~\\ref{sec:sampling}). Budget: 40 min of integration per "
            "run in the first rounds, 60 min afterwards and for the re-runs of the points that had exhausted it."
        ),
    )


# ------------------------------------------------------------------------------------------------ substitutos
PARITY_OUTPUTS = ("qL_top", "qG_top", "q_inj", "qL_max", "q_inj_max")  # médias e máximos (pedido do dono: não os spans)


def surrogate_parity(out, png):
    """Paridade no conjunto de teste, NN (cima) e PINN (baixo): as três vazões médias e os dois máximos."""
    res = _results()
    d = np.load(SURR / "data.npz", allow_pickle=True) if (SURR / "data.npz").exists() else None
    fig, axs = plt.subplots(2, len(PARITY_OUTPUTS), figsize=(fl.DOUBLE, 0.46 * fl.DOUBLE))
    if res is None or d is None:
        for a in axs.ravel():
            fl.placeholder(a, "surrogate results not found")
        fl.save(fig, out, "surrogate_parity", png)
        return
    X, Y, te = d["X"], d["Y"], d["test_idx"]
    names = res["outputs"]
    cols = [names.index(n) for n in PARITY_OUTPUTS if n in names]
    for i, k in enumerate(("nn", "pinn")):
        m = _model(k)
        Yp = _predict(m, X[te])
        for jj, j in enumerate(cols):
            ax = axs[i, jj]
            y, yp = Y[te, j], Yp[:, j]
            ax.scatter(
                y,
                yp,
                s=9,
                c=fl.COLORS.get("stratified", "tab:blue") if k == "nn" else fl.COLORS.get("annular", "tab:red"),
                edgecolor="k",
                lw=0.25,
            )
            lo, hi = min(y.min(), yp.min()), max(y.max(), yp.max())
            ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.6, ls="--")
            ax.set_title(
                f"{OUTPUT_LABELS[names[j]]}\n{k.upper()}, $R^2$ = {res['full'][k]['test']['r2'][j]:.2f}", fontsize=6.5
            )
            ax.tick_params(labelsize=6)
            if i == 1:
                ax.set_xlabel("simulated (kg s$^{-1}$)", fontsize=7)
            if jj == 0:
                ax.set_ylabel("predicted", fontsize=7)
    fig.tight_layout()
    fl.save(fig, out, "surrogate_parity", png)


def surrogate_learning(out, png):
    """Curvas de aprendizagem: RMSE de teste contra o número de pontos de treino, NN e PINN (média ± desvio), e a perda
    de treino e de teste por época das duas redes (pedido do dono: o conjunto de teste à vista no treino)."""
    res = _results()
    fig, axs = plt.subplots(1, 4, figsize=(fl.DOUBLE, 0.26 * fl.DOUBLE))
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
                v = (
                    [r[k]["test"]["rmse"][j] for r in sp if r["n_train"] == n]
                    if n != res["n_train"]
                    else [res["full"][k]["test"]["rmse"][j]]
                )
                mean.append(np.mean(v))
                sd.append(np.std(v))
            ax.errorbar(sizes, mean, yerr=sd, marker="o", ms=3, color=c, label=k.upper(), capsize=2, lw=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("training points")
        ax.set_ylabel(f"test RMSE, {OUTPUT_LABELS[name]} (kg s$^{{-1}}$)")
        ax.legend(fontsize=6.5)
    # (d) perda de dados no treino e perda no teste (EQM padronizado) por época, das histórias gravadas
    ax = axs[3]
    for k, c in (("nn", fl.COLORS.get("stratified", "tab:blue")), ("pinn", fl.COLORS.get("annular", "tab:red"))):
        f = SURR / f"{k}_history.npz"
        if not f.exists():
            continue
        h = np.load(f)["history"]
        ep = np.arange(1, h.shape[0] + 1)
        ax.plot(ep, h[:, 0], color=c, lw=0.9, label=f"{k.upper()} train")
        ax.plot(ep, h[:, 2], color=c, lw=0.9, ls="--", label=f"{k.upper()} test")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("standardised MSE")
    ax.legend(fontsize=6)
    for i, a in enumerate(axs):
        fl.panel_label(a, f"({'abcd'[i]})")
    fig.tight_layout()
    fl.save(fig, out, "surrogate_learning", png)


def surrogate_maps(out, png, model="pinn", name="surrogate_maps"):
    """Mapas do substituto (`model`: pinn ou nn) no plano das decisões (produção média de líquido, máximo de líquido,
    gás de lift) com os dados. A variante NN (`surrogate_maps_nn`, pedido do dono) serve de comparação."""
    d = np.load(SURR / "data.npz", allow_pickle=True) if (SURR / "data.npz").exists() else None
    mp = np.load(SURR / f"{model}_map.npz") if (SURR / f"{model}_map.npz").exists() else None
    fig, axs = plt.subplots(1, 3, figsize=(fl.DOUBLE, 0.34 * fl.DOUBLE))
    if d is None or mp is None:
        for a in axs:
            fl.placeholder(a, f"{model.upper()} maps not found")
        fl.save(fig, out, name, png)
        return
    X, Y = d["X"], d["Y"]
    # curvas de nível (pedido do dono), com os pontos simulados só como posições
    names = list(mp["names"]) if "names" in mp.files else list(_results()["outputs"])
    j_max = names.index("qL_max") if "qL_max" in names else 3
    specs = (
        (axs[0], 0, r"$\bar q_L$ at the top (kg s$^{-1}$)", np.arange(3.0, 5.51, 0.25), "%.2f"),
        (axs[1], j_max, r"max. $q_L$ at the top (kg s$^{-1}$)", np.array([5.0, 10.0, 20.0, 30.0, 40.0]), "%.0f"),
        (axs[2], 2, r"$\bar q_{gl}$ (kg s$^{-1}$)", np.arange(0.1, 1.21, 0.1), "%.1f"),
    )
    for ax, j, lab, levels, fmt in specs:
        Z = np.maximum(mp["Y"][:, :, j], 0.0) if j == j_max else mp["Y"][:, :, j]
        cs = ax.contour(mp["P"], mp["a"], Z, levels=levels, colors=fl.INK, linewidths=0.7)
        ax.clabel(cs, fmt=fmt, fontsize=6, inline=True, inline_spacing=2)
        ax.scatter(X[:, 1], X[:, 0], s=5, c="0.55", edgecolor="none", zorder=0)
        ax.set_xlabel("$P_{gas}$ (bar)")
        ax.set_title(lab, fontsize=8)
    axs[0].set_ylabel("choke opening $a$")
    for i, a in enumerate(axs):
        fl.panel_label(a, f"({'abc'[i]})")
    fig.tight_layout()
    fl.save(fig, out, name, png)


def surrogate_maps_nn(out, png):
    """Fig. 4 alternativa: os mesmos mapas com a rede só de dados."""
    surrogate_maps(out, png, model="nn", name="surrogate_maps_nn")


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
    rr = [
        f"residuals $R_1$--$R_{{{len(res['full']['pinn']['residual_rms'])}}}$ (RMS, std units)",
        "--",
        ", ".join(f"{v:.3f}" for v in res["full"]["nn"]["residual_rms"]),
        "--",
        ", ".join(f"{v:.3f}" for v in res["full"]["pinn"]["residual_rms"]),
    ]
    rows.append(rr)
    fl.tex_table(
        out,
        "tab_surrogate",
        ["output", "NN $R^2$", "NN RMSE", "PINN $R^2$", "PINN RMSE"],
        rows,
        align="lrrrr",
        notes=(
            f"Test set of {res['n_test']} runs ({res['n_train']} for training); RMSE in kg/s for the flow rates and bar for the "
            "pressures. Residuals: root-mean-square of the scaled physics residuals (the four balances and the four inequalities of the maxima) on 400 held-out collocation points. "
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
        [
            "training points",
            "$\\bar q_L$ NN",
            "$\\bar q_L$ PINN",
            "$\\bar q_G$ NN",
            "$\\bar q_G$ PINN",
            "$\\bar q_{gl}$ NN",
            "$\\bar q_{gl}$ PINN",
        ],
        rows,
        align="rrrrrrr",
        notes="Test RMSE (kg/s), mean $\\pm$ standard deviation over three random subsets of the training set; last row: the full training set.",
    )


def tab_cost(out):
    """Custo computacional (pedido do dono, 2026-10-20): compilação e integração por corrida (média, mínimo, máximo,
    por desfecho), treino e avaliação do substituto, e o que a mesma otimização custaria no simulador."""
    rows = _rows()
    res = _results()
    opt = json.load(open(OPT / "results.json")) if (OPT / "results.json").exists() else None
    if not rows or res is None or opt is None or "timing" not in opt:
        return
    groups = (
        ("all runs", rows),
        ("settled steady", [r for r in rows if r["reason"] == "steady"]),
        ("settled on a cycle", [r for r in rows if r["reason"] == "cycle"]),
        ("budget exhausted", [r for r in rows if r["reason"] == "failure"]),
    )
    comp = np.array([float(r["compile"]) for r in rows])
    body = [
        [
            "compilation of the model (first run of a process; s)",
            (
                f"{comp[comp > 1.0].mean():.0f}, {comp[comp > 1.0].min():.0f}--{comp[comp > 1.0].max():.0f}"
                if (comp > 1.0).any()
                else "--"
            ),
        ]
    ]
    for lab, sel in groups:
        cpu = np.array([float(r["cpu"]) for r in sel]) / 60.0
        body.append(
            [f"integration per run, {lab} ({len(sel)}; min)", f"{cpu.mean():.1f}, {cpu.min():.2g}--{cpu.max():.0f}"]
        )
    cpu_all = np.array([float(r["cpu"]) for r in rows])
    settled = [r for r in rows if r["reason"] != "failure"]
    t_run = np.mean([float(r["cpu"]) for r in settled])  # s: corrida assentada média
    body.append(["campaign, all runs (h)", f"{cpu_all.sum() / 3600:.0f}"])
    tm = opt["timing"]
    body.append(
        [
            "surrogate training, 4000 epochs (s)",
            f"{res['full']['pinn']['wall']:.1f} (PINN), {res['full']['nn']['wall']:.1f} (NN)",
        ]
    )
    body.append(["one evaluation of the surrogate (ms)", f"{tm['single_eval_s'] * 1e3:.2f}"])
    body.append([f"grid of {tm['grid_eval_n']} evaluations in one batch (s)", f"{tm['grid_eval_s']:.3f}"])
    n_opt = tm["n_eval_optimise"]["grid"] + tm["n_eval_optimise"]["refine"]
    n_cv = tm["n_eval_curve"]["grid"] + tm["n_eval_curve"]["refine"]
    body.append([f"optimisation, {n_opt} evaluations (s)", f"{tm['optimise_s']:.2f}"])
    body.append([f"operating curve, {n_cv} evaluations (s)", f"{tm['curve_s']:.2f}"])
    n_tot = n_opt + n_cv
    body.append(
        ["same evaluations on the simulator, mean settled run (years)", f"{n_tot * t_run / 3600 / 24 / 365:.1f}"]
    )
    body.append(["derivative-free optimisation on the simulator, 200 runs (h)", f"{200 * t_run / 3600:.0f}"])
    fl.tex_table(
        out,
        "tab_cost",
        ["", ""],
        body,
        align="lr",
        notes=(
            "Wall-clock times of the campaigns (JAX backend; each run in one process of the campaign, which XLA spreads over "
            "three to four cores of a laptop) and of the surrogate on the same machine, so that the hours are process-hours; "
            "the model compiles once per process and the compiled code serves the later runs of that process. The simulator equivalents multiply the number of evaluations of the optimisation "
            "and of the operating curve of Section~\\ref{sec:optresults} by the mean integration time of a settled run; "
            "the last row is a direct derivative-free search on the simulator for one set of prices and capacities."
        ),
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


OPT = fl.RUNS / "paper02" / "optimize"
# ótimos com restrições (pedido do dono): pasta → rótulo (capacidade do compressor em kg/s; "sep." = separador)
CONSTRAINED = (
    ("optimize_gl15", "1.5"),
    ("optimize_gl1", "1.0"),
    ("optimize_constrained", "0.8"),
    ("optimize_gl05", "0.5"),
    ("optimize_gl05_sep6", "0.5, sep. 6"),
)
GL_LEVELS = (1.75,)  # curva de nível do pico de gás de lift previsto [kg/s]: a capacidade do compressor do exemplo
QL_LEVELS = (20.0,)  # e do pico de líquido [kg/s]: a capacidade do separador do exemplo


def fob_map(out, png):
    """Função objetivo (receita do óleo − compressão) do substituto no plano das decisões, curvas de nível, com o
    ótimo e as corridas por classe (a fronteira de estabilidade dos dados)."""
    res = json.load(open(OPT / "results.json")) if (OPT / "results.json").exists() else None
    mp = np.load(OPT / "fob_map.npz") if (OPT / "fob_map.npz").exists() else None
    rows = _rows()
    fig, axs = plt.subplots(1, 3, figsize=(fl.DOUBLE, 0.4 * fl.DOUBLE), gridspec_kw=dict(width_ratios=[1.6, 1.0, 1.0]))
    if res is None or mp is None:
        for a in axs:
            fl.placeholder(a, "optimisation results not found (runs/paper02/optimize)")
        fl.save(fig, out, "fob_map", png)
        return
    F = mp["fob"] / 1e3
    levels = np.linspace(np.floor(F.min() / 10) * 10, np.ceil(F.max() / 10) * 10, 13)
    cs = axs[0].contour(mp["P"], mp["a"], F, levels=levels, colors=fl.INK, linewidths=0.7)
    axs[0].clabel(cs, fmt="%.0f", fontsize=6, inline=True)
    if rows:
        P = np.array([float(r["P_gas_bar"]) for r in rows])
        A = np.array([float(r["choke_opening"]) for r in rows])
        cls = np.array([r["class"] for r in rows])
        for c, col in CLASS_COLORS.items():
            m = cls == c
            if m.any():  # uma classe sem pontos não entra na legenda
                axs[0].scatter(P[m], A[m], c=col, s=7, edgecolor="none", label=c, zorder=2)
    axs[0].plot(
        [res["P_gas_bar"]],
        [res["a"]],
        marker="*",
        ms=11,
        color=fl.COLORS.get("annular", "tab:red"),
        mec="k",
        mew=0.5,
        zorder=5,
        label="optimum (example)",
    )
    # restrições: região inadmissível pintada (pedido do dono) — pico de gás de lift acima da capacidade do
    # compressor, pico de líquido acima da do separador —, as curvas de nível dos dois limites e os ótimos
    from matplotlib.patches import Patch

    c_gl, c_ql = fl.COLORS.get("annular", "tab:red"), fl.COLORS.get("stratified", "tab:blue")
    if "q_gl_max" in mp:
        over = (mp["q_gl_max"] > GL_LEVELS[0]).astype(float)
        axs[0].contourf(mp["P"], mp["a"], over, levels=[0.5, 1.5], colors=[c_gl], alpha=0.12, zorder=0)
    if "qL_max" in mp:
        over = (mp["qL_max"] > QL_LEVELS[0]).astype(float)
        axs[0].contourf(mp["P"], mp["a"], over, levels=[0.5, 1.5], colors=[c_ql], alpha=0.12, zorder=0)
    if "q_gl_max" in mp:
        cg = axs[0].contour(
            mp["P"],
            mp["a"],
            mp["q_gl_max"],
            levels=GL_LEVELS,
            colors=fl.COLORS.get("annular", "tab:red"),
            linewidths=0.8,
            linestyles="--",
        )
        axs[0].clabel(cg, fmt=lambda v: f"$q_{{gl}}$ {v:g}", fontsize=5.5, inline=True)
    if "qL_max" in mp:
        cq = axs[0].contour(
            mp["P"],
            mp["a"],
            mp["qL_max"],
            levels=QL_LEVELS,
            colors=fl.COLORS.get("stratified", "tab:blue"),
            linewidths=0.8,
            linestyles=":",
        )
        axs[0].clabel(cq, fmt=lambda v: f"$q_L$ {v:g}", fontsize=5.5, inline=True)
    first = True
    for folder, label in CONSTRAINED:
        f = fl.RUNS / "paper02" / folder / "results.json"
        if not f.exists():
            continue
        rc = json.load(open(f))
        axs[0].plot(
            [rc["P_gas_bar"]],
            [rc["a"]],
            marker="s",
            ms=5,
            color="w",
            mec="k",
            mew=0.8,
            zorder=6,
            label="other compressors (kg/s)" if first else None,
        )
        axs[0].annotate(
            label, (rc["P_gas_bar"], rc["a"]), xytext=(3, -8), textcoords="offset points", fontsize=5.5, zorder=6
        )
        first = False
    axs[0].set_xlabel("$P_{gas}$ (bar)")
    axs[0].set_ylabel("choke opening $a$")
    handles, labels = axs[0].get_legend_handles_labels()
    handles += [
        Patch(facecolor=c_gl, alpha=0.25, label=f"peak $q_{{gl}}$ > {GL_LEVELS[0]:g} kg/s"),
        Patch(facecolor=c_ql, alpha=0.25, label=f"peak $q_L$ > {QL_LEVELS[0]:g} kg/s"),
    ]
    labels += [h.get_label() for h in handles[-2:]]
    # legenda única no topo da figura (os painéis não têm título: a legenda da figura diz o que são)
    fig.legend(
        handles,
        labels,
        fontsize=6,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=7,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    # curva operacional (pedido do dono, 2026-10-20): a P_gas fixa, a abertura que maximiza a FOB entre as
    # admissíveis (tools/paper02_optimize.py, operating_curve.npz); a faixa de pressões sem abertura admissível fica
    # pintada, e a estrela é o ótimo do exemplo
    cv = np.load(OPT / "operating_curve.npz") if (OPT / "operating_curve.npz").exists() else None
    c_opt = fl.COLORS.get("annular", "tab:red")
    if cv is None:
        for ax in axs[1:]:
            fl.placeholder(ax, "operating curve not found (runs/paper02/optimize)")
    else:
        ok = cv["feasible"].astype(bool)
        Pc = cv["P_gas_bar"]
        for ax in axs[1:]:
            if (~ok).any():
                ax.axvspan(Pc[~ok].min(), Pc[~ok].max(), color=c_gl, alpha=0.12, lw=0)
        axs[1].plot(Pc[ok], cv["a"][ok], color=fl.INK, lw=1.2)
        axs[1].plot([res["P_gas_bar"]], [res["a"]], marker="*", ms=9, color=c_opt, mec="k", mew=0.5, zorder=5)
        axs[1].set_ylim(0.0, 1.05)
        axs[1].set_ylabel("optimal choke opening $a^*$")
        axs[2].plot(Pc[ok], cv["revenue"][ok] / 1e3, color=fl.INK, lw=1.0, label="oil revenue")
        axs[2].plot(Pc[ok], cv["cost"][ok] / 1e3, color=c_opt, lw=1.0, label="compression cost")
        axs[2].plot(
            Pc[ok],
            cv["fob"][ok] / 1e3,
            color=fl.COLORS.get("stratified", "tab:blue"),
            lw=1.2,
            ls="--",
            label="objective",
        )
        axs[2].plot([res["P_gas_bar"]], [res["fob"] / 1e3], marker="*", ms=9, color=c_opt, mec="k", mew=0.5, zorder=5)
        axs[2].set_ylabel(r"$10^3$ \$/d")
        axs[2].legend(fontsize=6.5, loc="center")
    for ax in axs[1:]:
        ax.set_xlabel("$P_{gas}$ (bar)")
    for k, ax in enumerate(axs):
        fl.panel_label(ax, f"({'abc'[k]})")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fl.save(fig, out, "fob_map", png)


FIGURES = {
    "system": system,
    "fob_map": fob_map,
    "dataset_map": dataset_map,
    "surrogate_parity": surrogate_parity,
    "surrogate_learning": surrogate_learning,
    "surrogate_maps": surrogate_maps,
    "surrogate_maps_nn": surrogate_maps_nn,
}
TABLES = {"tab_dataset": tab_dataset, "tab_surrogate": tab_surrogate, "tab_sparse": tab_sparse, "tab_cost": tab_cost}

if __name__ == "__main__":
    fl.run_cli(FIGURES, __doc__.splitlines()[0], tables=TABLES)
