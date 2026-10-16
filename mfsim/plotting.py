"""Gráficos no estilo das figuras do capítulo 10 da tese."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

LABELS = {
    "alphaG": "Fração de Área de Gás (m³/m³)",
    "P_bar": "Pressão (10⁵ Pa)",
    "qL": "Vazão de Líquido (kg/s)",
    "qG": "Vazão de Gás (kg/s)",
    "velocidadeL": "Velocidade do Líquido (m/s)",
    "velocidadeG": "Velocidade do Gás (m/s)",
}


def surface(ax, x, t, Z, zlabel="", title="", cmap="jet", stride=1, elev=25, azim=-135):
    """Superfície 3D (posição x tempo), como nas figuras 10.3, 10.11, 10.16 etc."""
    X, T = np.meshgrid(x, t)
    ax.plot_surface(T, X, Z.T, cmap=cmap, rstride=stride, cstride=stride, linewidth=0, antialiased=True)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Posição (m)")
    ax.set_zlabel(zlabel, labelpad=6)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.view_init(elev=elev, azim=azim)
    return ax


def pipe_surfaces(res, k=0, keys=("alphaG", "P_bar", "qL", "qG"), title=None, figsize=(11, 8), **kw):
    """Quatro superfícies de um duto (fração de gás, pressão, vazões)."""
    p = res.pipe(k)
    fig = plt.figure(figsize=figsize)
    title = title or f"Tubulação - {k + 1}"
    for i, key in enumerate(keys):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        surface(ax, p["x"], res.T, p["out"][key], zlabel=LABELS.get(key, key), title=title, **kw)
    fig.tight_layout()
    return fig


def pipe_profiles(res, k=0, keys=("P_bar", "alphaG"), it=0, legacy=None, figsize=(10, 3.6)):
    """Perfis ao longo do duto num instante (ex.: condição inicial, Fig. 10.2)."""
    p = res.pipe(k)
    fig, axes = plt.subplots(1, len(keys), figsize=figsize)
    for ax, key in zip(np.atleast_1d(axes), keys):
        ax.plot(p["x"], p["out"][key][:, it], "b-", label="mfsim (Python)")
        if legacy is not None and key in legacy:
            ax.plot(legacy["x"], legacy[key][:, it], "r--", label="MFSIM (MATLAB)")
        ax.set_xlabel("Posição (m)")
        ax.set_ylabel(LABELS.get(key, key))
        ax.set_title(f"Tubulação - {k + 1}", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
    if legacy is not None:
        np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def pipe_timeseries(
    res, k=0, keys=("alphaG", "P_bar", "qL", "qG"), nodes=None, legacy=None, figsize=(11, 7), labels=None
):
    """Séries temporais em alguns nós (comparação Python x MATLAB)."""
    p = res.pipe(k)
    n = p["x"].size
    if nodes is None:
        nodes = [0, n // 2, n - 1]
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, key in zip(axes.ravel(), keys):
        for j, node in enumerate(nodes):
            lab = f"x={p['x'][node]:.0f} m"
            ax.plot(res.T, p["out"][key][node], color=f"C{j}", label=lab)
            if legacy is not None and key in legacy:
                jl = int(np.argmin(np.abs(legacy["x"] - p["x"][node])))
                ax.plot(legacy["T"], legacy[key][jl], "--", color=f"C{j}", alpha=0.8)
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel(LABELS.get(key, key))
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, title="— Python  -- MATLAB" if legacy is not None else None)
    fig.tight_layout()
    return fig


def compare_runs(runs, k=0, key="alphaG", node=-1, ax=None, title=None, legacy=None):
    """Compara várias simulações (ex.: sensibilidade de malha, Fig. 10.4) num nó."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for label, res in runs.items():
        p = res.pipe(k)
        ax.plot(res.T, p["out"][key][node], label=label)
    if legacy is not None and key in legacy:
        ax.plot(legacy["T"], legacy[key][node], "k--", label="MATLAB (tese)")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel(LABELS.get(key, key))
    ax.set_title(title or "", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def eigenvalue_plot(ev, ax=None, title=""):
    """Autovalores da Jacobiana no plano complexo (Figs. 10.7 e 10.8)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ev.real, ev.imag, "b.", ms=4)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Parte real")
    ax.set_ylabel("Parte imaginária")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    n_pos = int(np.sum(ev.real > 0))
    ax.text(
        0.02, 0.95, f"{ev.size} autovalores, {n_pos} com parte real > 0", transform=ax.transAxes, fontsize=8, va="top"
    )
    return ax


def lab_validation_figure(run, tag, t0=None, t1=None, out=None, dpi=90):
    """Figura rápida de uma corrida de validação de laboratório: pressão na entrada (kPa) com
    as faixas publicadas, vazão de líquido na saída, fração de gás na base e no topo do riser
    e velocidades das fases na base. (Era tools/val_plot.py.)"""
    from .analysis import lab_reference

    case = run.case
    fluid = case.fluid
    m = case.meta
    A = np.pi * case.pipes[0].D ** 2 / 4
    qL = m["qL"]
    T = run.T
    p = run.pipe(0)
    x = p["x"]
    ib = int(np.argmin(np.abs(x - m["s_elbow0"])))
    it_ = int(np.argmin(np.abs(x - (m["s_elbow0"] + m["H_riser"]))))
    t0 = T[0] if t0 is None else t0
    t1 = T[-1] if t1 is None else t1
    sel = (T >= t0) & (T <= t1)
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(T[sel], p["out"]["P_bar"][1][sel] * 100)
    ax[0].set_ylabel("P entrada (kPa)")
    key, ref = lab_reference(tag)
    if key:
        ax[0].axhline(ref["P_min"], ls=":", c="k")
        ax[0].axhline(ref["P_max"], ls=":", c="k")
        ax[0].set_title(ref["fonte"] + f" | exp. período {ref['periodo']} s", fontsize=9)
    ax[1].plot(T[sel], p["out"]["qL"][-1][sel])
    ax[1].axhline(qL, ls=":", c="k")
    ax[1].set_ylabel("qL saída (kg/s)")
    ax[2].plot(T[sel], p["out"]["alphaG"][ib + 1][sel], label="base do riser")
    ax[2].plot(T[sel], p["out"]["alphaG"][it_ - 1][sel], label="topo do riser")
    ax[2].plot(T[sel], p["out"]["alphaG"][ib - 2][sel], label="linha, 2 células antes do joelho")
    ax[2].legend(fontsize=8)
    ax[2].set_ylabel("αG")
    vG_ = p["out"]["qG"][ib + 1] / (
        fluid.rhoG(p["out"]["P_bar"][ib + 1] * 1e5) * np.maximum(p["out"]["alphaG"][ib + 1], 1e-3) * A
    )
    vL_ = p["out"]["qL"][ib + 1] / (fluid.rhoL0 * np.maximum(1 - p["out"]["alphaG"][ib + 1], 1e-3) * A)
    ax[3].plot(T[sel], vG_[sel], label="vG base riser")
    ax[3].plot(T[sel], vL_[sel], label="vL base riser")
    ax[3].set_ylabel("m/s")
    ax[3].legend(fontsize=8)
    ax[3].set_xlabel("t (s)")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=dpi)
    return fig


# ----------------------------------------------------------------------------- mapa de classes (issue 043)
CYCLE_COLORS = {"stable": "#2a78d6", "cycling": "#1baf7a", "slugging": "#eb6834", "fail": "#e1e0d9"}
CYCLE_LABELS = {
    "stable": "Estável: escoamento permanente",
    "cycling": "Ciclo de pressão, saída suave",
    "slugging": "Golfada severa: pico na saída ≥ 2× a média",
    "fail": "Corrida falhou",
}


def _cell_edges(v, log=False):
    v = np.log(v) if log else np.asarray(v, float)
    mid = 0.5 * (v[1:] + v[:-1])
    e = np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]])
    return np.exp(e) if log else e


def cycle_map(ax, rows, x="ab", y="P_res", logx=True, logy=True, annotate=True, best="q_mean", surface="#fcfcfb"):
    """Mapa de classes de ciclo (`mfsim.analysis.cycle_class`) numa grelha de dois parâmetros.

    `rows`: dicts com `x`, `y`, `cls` (e `success`); com `annotate`, cada célula leva a pressão
    média (estável) ou a pressão máxima e o período (com ciclo), com vírgula decimal; com `best`,
    a célula estável de maior `best` de cada linha fica contornada. Devolve as arestas (xe, ye)."""
    from matplotlib.patches import Rectangle

    xs = sorted({r[x] for r in rows})
    ys = sorted({r[y] for r in rows})
    xe, ye = _cell_edges(xs, log=logx), _cell_edges(ys, log=logy)
    G = {(r[y], r[x]): r for r in rows}
    for i, yv in enumerate(ys):
        top = None
        for j, xv in enumerate(xs):
            r = G.get((yv, xv))
            if r is None:
                continue
            cls = r["cls"] if r.get("success", True) else "fail"
            ax.add_patch(
                Rectangle(
                    (xe[j], ye[i]), xe[j + 1] - xe[j], ye[i + 1] - ye[i], fc=CYCLE_COLORS[cls], ec=surface, lw=1.2
                )
            )
            if annotate:
                if cls == "stable":
                    txt = f"{r['P_mean']:.1f}"
                else:
                    per = r.get("period", np.nan)
                    if np.isfinite(per):
                        txt = f"{r['P_max']:.1f}\n" + (f"{per / 60:.0f} min" if per >= 120 else f"{per:.0f} s")
                    else:
                        txt = f"{r['P_max']:.1f}"
                xc, yc = (np.sqrt(xe[j] * xe[j + 1]) if logx else 0.5 * (xe[j] + xe[j + 1])), (
                    np.sqrt(ye[i] * ye[i + 1]) if logy else 0.5 * (ye[i] + ye[i + 1])
                )
                ax.text(xc, yc, txt.replace(".", ","), ha="center", va="center", fontsize=6.0, color="#0b0b0b")
            if best and cls == "stable" and (top is None or r[best] > top[0]):
                top = (r[best], j)
        if best and top is not None:
            j = top[1]
            ax.add_patch(
                Rectangle(
                    (xe[j], ye[i]), xe[j + 1] - xe[j], ye[i + 1] - ye[i], fc="none", ec="#0b0b0b", lw=1.8, zorder=5
                )
            )
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{v:g}".replace(".", ",") for v in ys])
    ax.minorticks_off()
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    for s in ax.spines.values():
        s.set_visible(False)
    return xe, ye
