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


def pipe_timeseries(res, k=0, keys=("alphaG", "P_bar", "qL", "qG"), nodes=None, legacy=None,
                    figsize=(11, 7), labels=None):
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


def legacy_pipe_dict(raw, k=0):
    """Extrai de `saidas.tubo{k}` um dict compatível com os gráficos acima."""
    from .io_legacy import legacy_pipe_outputs, legacy_states
    out = legacy_pipe_outputs(raw, k)
    T, X = legacy_states(raw)
    eixos = raw.get("eixos", {})
    try:
        xs = eixos["tubo"]["X"]["valor"]
        x = np.asarray(xs[k] if isinstance(xs, list) else xs, float).ravel()
    except Exception:
        x = np.arange(out["alfaG"].shape[0], dtype=float)
    d = {"T": T, "x": x}
    d["alphaG"] = np.asarray(out["alfaG"], float)
    d["P_bar"] = np.asarray(out["pressao"], float)
    d["qL"] = np.asarray(out.get("vazaoMassaL", np.zeros_like(d["alphaG"])), float)
    d["qG"] = np.asarray(out.get("vazaoMassaG", np.zeros_like(d["alphaG"])), float)
    for key in ("velocidadeL", "velocidadeG", "reg1"):
        if key in out:
            d[key] = np.asarray(out[key], float)
    return d


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
    ax.text(0.02, 0.95, f"{ev.size} autovalores, {n_pos} com parte real > 0", transform=ax.transAxes,
            fontsize=8, va="top")
    return ax
