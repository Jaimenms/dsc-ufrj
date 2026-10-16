"""Figuras e tabelas do artigo 01 (intermitência severa com o modelo a dois fluidos e padrão de escoamento
dinâmico) a partir das corridas gravadas e de caches calculados do modelo sem integrar no tempo; não simula.

uso: uv run python figures/make_figures.py --out figures/generated --tables tables/generated [--only NOME] [--no-png]
     (a partir da pasta do artigo; `make figures` faz o mesmo)

Corridas (ver PLAN.md): `runs/maps/well_auto/` (mapa, `tools/slug_map.py` + `mfsim batch`),
`runs/paper01/<lote>/` (`mfsim batch batches/paper01_*.json`) e `notebooks/data/severe_slug_map_well.npz`
(mapa com padrão fixo, issue 043). Os caches (`runs/paper01/cache/`: mapas de hiperbolicidade, diagnósticos
de trajetória) refazem-se sozinhos quando faltam. Sem corridas, a figura sai com um aviso e a tabela com «--»,
para o PDF compilar na mesma.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_common"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figlib as fl  # noqa: E402

PAPER = "paper01"
RUNS = fl.RUNS / PAPER
CACHE = RUNS / "cache"
BASE_CASE = fl.PROJECT / "cases" / "severe_slug" / "well_choke_auto.json"
MAP_AUTO = fl.RUNS / "maps" / "well_auto"
PATTERN_LABEL = {"stratified": "stratified", "intermittent": "intermittent", "bubble": "bubbly", "annular": "annular"}


def _cached(name, compute):
    """Dicionário de arrays em `CACHE/name.npz`, calculado por `compute()` se não existe."""
    path = CACHE / f"{name}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    out = compute()
    if "missing" not in out:  # sem as corridas de que depende: não grava (recalcula quando chegarem)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, **out)
    return out


def _base_network():
    from mfsim.io import case_from_json

    return case_from_json(str(BASE_CASE))


# ------------------------------------------------------------------------------------------------ geometria
def geometry(out, png):
    """Traçado do caso base: linha, descendente, curva em arco e riser, com a malha (centros das células) e os
    elementos de contorno (reservatório com IPR, choke, separador); detalhe da curva à escala."""
    from mfsim.geometry import pipe_mesh

    net = _base_network()
    pipe = net.pipes[0]
    pts = np.asarray(pipe.points, float)
    mesh = pipe_mesh(pts[:, 0], pts[:, 1], pts[:, 2], np.asarray(pipe.n_elements, int), grading=pipe.grading)
    y, z = mesh["xyz"][:, 1], mesh["xyz"][:, 2]
    yc, zc = 0.5 * (y[1:] + y[:-1]), 0.5 * (z[1:] + z[:-1])
    fig = plt.figure(figsize=(fl.DOUBLE, 0.36 * fl.DOUBLE))
    ax = fig.add_axes([0.07, 0.17, 0.62, 0.78])
    ax.plot(y, z, color=fl.INK, lw=1.4, solid_capstyle="round")
    ax.plot(yc, zc, "o", ms=2.4, color=fl.COLORS["stratified"], label="cell centres")
    ax.plot(y, z, "|", ms=5, color=fl.INK2, mew=0.6)
    ax.annotate(
        "reservoir\n(linear IPR)",
        xy=(y[0], z[0]),
        xytext=(y[0] + 8, z[0] + 9),
        fontsize=7,
        arrowprops=dict(arrowstyle="-", lw=0.5, color=fl.INK2),
    )
    ax.annotate(
        "choke",
        xy=(y[-1], z[-1]),
        xytext=(y[-1] - 70, z[-1] + 4),
        fontsize=7,
        arrowprops=dict(arrowstyle="-", lw=0.5, color=fl.INK2),
    )
    ax.text(y[-1] - 70, z[-1] + 0.8, "separator 4 bar", fontsize=7, color=fl.INK2)
    m = net.meta or {}
    ax.text(150, -4.0, f"downward line, {m.get('slope_down_deg', -3):g}°", fontsize=7, rotation=-6, color=fl.INK2)
    ax.text(y[-1] - 3, 0.5 * (z.min() + z.max()), "riser", fontsize=7, rotation=90, va="center", ha="right")
    ax.set_xlabel("horizontal distance (m)")
    ax.set_ylabel("elevation (m)")
    ax.set_xlim(-10, y[-1] + 15)
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=7)
    fl.panel_label(ax, "(a)")
    # detalhe da curva à escala
    ax2 = fig.add_axes([0.76, 0.17, 0.22, 0.78])
    sel = (y > y[-1] - 25) & (z < z.min() + 18)
    ax2.plot(y[sel], z[sel], color=fl.INK, lw=1.4)
    k = (yc > y[-1] - 25) & (zc < z.min() + 18)
    ax2.plot(yc[k], zc[k], "o", ms=3, color=fl.COLORS["stratified"])
    for yi, zi, th in zip(yc[k], zc[k], np.rad2deg(mesh["theta"])[k]):
        if 1 < th < 89:
            ax2.text(yi + 0.6, zi - 0.4, f"{th:.0f}°", fontsize=6, color=fl.INK2)
    ax2.set_aspect("equal")
    ax2.set_xlabel("horizontal distance (m)")
    ax2.set_title("bend, true scale", fontsize=7)
    ax2.grid(True)
    fl.panel_label(ax2, "(b)")
    fl.save(fig, out, "geometry", png)


# ------------------------------------------------------------------------------------------------ boa colocação
HYP_PRESSURE, HYP_VL = 6e5, 0.5  # Pa e m/s: estado típico da base do riser no caso base
HYP_ALPHA = np.logspace(-3, np.log10(0.995), 70)
HYP_SLIP = np.linspace(-1.0, 6.0, 71)
HYP_PANELS = (
    # (padrão, regime fixo, ângulo, regularização (dict de PipeParams), rótulo da regularização)
    ("stratified", 4, -3.0, dict(low_void_regime=2, virtual_mass=0.5), "low-void rule + virtual mass"),
    ("intermittent", 3, 90.0, dict(interfacial_pressure=1.2), r"interfacial pressure, $\delta = 1.2$"),
    ("bubble", 2, 90.0, dict(virtual_mass=0.5), r"virtual mass, $C_{vm} = 0.5$"),
    ("annular", 1, 90.0, dict(interfacial_pressure=1.2), r"interfacial pressure, $\delta = 1.2$"),
)


def _hyperbolicity_compute():
    from mfsim.models.base import PipeParams
    from mfsim.models.twofluid import TwoFluidModel

    fluid = _base_network().fluid
    D = 0.05
    A = np.pi * D**2 / 4
    rhoL, rhoG, _, _ = fluid.props(HYP_PRESSURE)
    out = {"alpha": HYP_ALPHA, "slip": HYP_SLIP}
    for name, regime, theta, reg, _ in HYP_PANELS:
        for tag, kw in (("raw", {}), ("reg", reg)):
            params = PipeParams(D=D, eps=1e-5, PR=1e6, regime=regime, bubble_drag="fisico", low_void_alpha=0.01, **kw)
            m = TwoFluidModel(fluid, params)
            Z = np.zeros((HYP_ALPHA.size, HYP_SLIP.size))
            for i, a in enumerate(HYP_ALPHA):
                y = np.array(
                    [
                        [a, HYP_PRESSURE / 1e6, rhoL * (1 - a) * A * HYP_VL, rhoG * a * A * (HYP_VL + s)]
                        for s in HYP_SLIP
                    ]
                )
                Z[i] = np.abs(m.eigenvalues(y, np.deg2rad(np.full(HYP_SLIP.size, theta))).imag).max(axis=1)
            out[f"{name}_{tag}"] = Z
    # massa virtual na forma simples (só a diferença das acelerações materiais, lambda = 2)
    m = TwoFluidModel(
        fluid,
        PipeParams(D=D, eps=1e-5, PR=1e6, regime=2, bubble_drag="fisico", virtual_mass=0.5, vm_lambda=2.0),
    )
    Z = np.zeros((HYP_ALPHA.size, HYP_SLIP.size))
    for i, a in enumerate(HYP_ALPHA):
        y = np.array(
            [[a, HYP_PRESSURE / 1e6, rhoL * (1 - a) * A * HYP_VL, rhoG * a * A * (HYP_VL + s)] for s in HYP_SLIP]
        )
        Z[i] = np.abs(m.eigenvalues(y, np.deg2rad(np.full(HYP_SLIP.size, 90.0))).imag).max(axis=1)
    out["bubble_lam2"] = Z
    return out


def hyperbolicity(out, png):
    """max |Im λ| (m/s) das velocidades características por fechamento, na fração de gás × escorregamento, sem
    (em cima) e com (em baixo) a regularização; branco = hiperbólico."""
    d = _cached("hyperbolicity", _hyperbolicity_compute)
    fig, axes = plt.subplots(2, 4, figsize=(fl.DOUBLE, 0.52 * fl.DOUBLE), sharex=True, sharey=True)
    vmax = 3.0
    letters = iter("abcdefgh")
    for j, (name, _, theta, _, reg_label) in enumerate(HYP_PANELS):
        for i, tag in enumerate(("raw", "reg")):
            ax = axes[i, j]
            Z = np.where(d[f"{name}_{tag}"] > 1e-6, d[f"{name}_{tag}"], np.nan)
            pc = ax.pcolormesh(
                d["slip"], d["alpha"], Z, shading="nearest", cmap="Oranges", vmin=0, vmax=vmax, rasterized=True
            )
            ax.set_yscale("log")
            frac = float(np.mean(d[f"{name}_{tag}"] > 1e-6))
            ax.text(
                0.97,
                0.04,
                f"ill-posed {100 * frac:.0f} %",
                transform=ax.transAxes,
                ha="right",
                fontsize=6.5,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0),
            )
            if i == 0:
                ax.set_title(f"{PATTERN_LABEL[name]} ({theta:g}°)", fontsize=7.5)
            else:
                ax.set_xlabel(r"slip $v_G - v_L$ (m s$^{-1}$)")
                ax.text(
                    0.03,
                    0.96,
                    reg_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0),
                )
            fl.panel_label(ax, f"({next(letters)})")
        axes[0, 0].set_ylabel(r"$\alpha_G$, as closed")
        axes[1, 0].set_ylabel(r"$\alpha_G$, regularised")
    cb = fig.colorbar(pc, ax=axes, shrink=0.85, pad=0.015)
    cb.set_label(r"max $|\mathrm{Im}\,\lambda|$ (m s$^{-1}$)")
    fl.save(fig, out, "hyperbolicity", png)


def tab_hyperbolicity(out):
    """Fração mal posta do mapa por fechamento, sem e com regularização."""
    d = _cached("hyperbolicity", _hyperbolicity_compute)
    rows = []
    for name, _, theta, _, reg_label in HYP_PANELS:
        rows.append(
            [
                f"{PATTERN_LABEL[name]} (${theta:g}^\\circ$)",
                fl.fmt(100 * np.mean(d[f"{name}_raw"] > 1e-6), "{:.0f}"),
                fl.fmt(float(np.max(d[f"{name}_raw"])), "{:.2f}"),
                reg_label,
                fl.fmt(100 * np.mean(d[f"{name}_reg"] > 1e-6), "{:.0f}"),
            ]
        )
    if "bubble_lam2" in d:
        rows.append(
            [
                "bubbly ($90^\\circ$)",
                fl.fmt(100 * np.mean(d["bubble_raw"] > 1e-6), "{:.0f}"),
                fl.fmt(float(np.max(d["bubble_raw"])), "{:.2f}"),
                "virtual mass, $\\lambda = 2$ (not objective)",
                fl.fmt(100 * np.mean(d["bubble_lam2"] > 1e-6), "{:.0f}"),
            ]
        )
    fl.tex_table(
        out,
        "tab_hyperbolicity",
        [
            "closure (inclination)",
            "ill-posed (\\%)",
            "max $|\\mathrm{Im}\\,\\lambda|$ (m/s)",
            "regularisation",
            "ill-posed (\\%)",
        ],
        rows,
        align="llrlr",
        notes=(
            f"Map of $70\\times71$ states: $10^{{-3}}\\le\\alpha_G\\le0.995$, $-1\\le v_G-v_L\\le 6$ m/s, "
            f"$v_L={HYP_VL:g}$ m/s, $P={HYP_PRESSURE / 1e5:g}$ bar, $D=50$ mm."
        ),
    )


# ------------------------------------------------------------------------------------------------ mapa e corridas
import os  # noqa: E402

MAP_DIR = Path(os.environ.get("PAPER01_MAP_DIR", str(fl.RUNS / "maps" / "well_auto")))
MAP_FIXED_DIR = fl.RUNS / "maps" / "well"  # padrão fixo (issue 043), para comparação
P_SEL = 15.0  # bar: linha do mapa das figuras de dinâmica
OPENINGS_SEL = (0.13, 0.28, 0.6)  # estável, ciclo de pressão, golfada severa (a confirmar no mapa dinâmico)
CLASS_LABEL = {"stable": "stable", "cycling": "pressure cycle, smooth outflow", "slugging": "severe slugging"}


def _name(ab, P):
    return f"ab{ab:.3f}_P{P:04.1f}"


def _map_rows(map_dir=None):
    """Linhas do resumo do mapa (`summary.json` de `tools/slug_map.py plot`) ou None."""
    path = Path(map_dir or MAP_DIR) / "summary.json"
    if not path.exists():
        return None
    return json.load(open(path))


def _map_run(start, ab, P, map_dir=None):
    from mfsim.runs import load_run

    path = Path(map_dir or MAP_DIR) / start / "runs" / f"{_name(ab, P)}.npz"
    return load_run(str(path)) if path.exists() else None


def _dense_run(ab, P=P_SEL):
    """Corrida densa do lote `paper01_dynamics` (saída a 1 s) ou, na falta dela, a do mapa (5 s)."""
    run = fl.load_run(f"paper01/dynamics/{_name(ab, P)}")
    return run if run is not None else _map_run("steady", ab, P)


def _steady_state(ab, P, map_dir=None):
    """Estado estacionário exato de um ponto do mapa (condição inicial do caso da partida do estacionário,
    gravada nos nós do esquema antes da perturbação): (modelo, X) ou None."""
    from mfsim.io import case_from_json
    from mfsim.simulation import build_model

    path = Path(map_dir or MAP_DIR) / "steady" / "cases" / f"{_name(ab, P)}.json"
    if not path.exists():
        return None
    net = case_from_json(str(path))
    if net.initial != "steady":
        return None
    model = build_model(net)
    return model, model.initial_state()


def _zones(model):
    """Índices de célula (0-based, células do esquema) do fim da linha, do ponto baixo, da base do riser e do
    topo, e as posições s das células."""
    sch = model.pipes[0].scheme
    th = np.rad2deg(sch.theta_nodes[1 : sch.N + 1])
    s = np.asarray(sch.x_cells)
    k_bend = np.flatnonzero((th > 1.0) & (th < 89.0))
    k_riser = np.flatnonzero(th > 89.0)
    return dict(s=s, th=th, k_bend=k_bend, k_riser=k_riser, s_low=float(s[k_bend[0]]) if k_bend.size else np.nan)


def _pattern_cmap():
    from matplotlib.colors import ListedColormap

    return ListedColormap([fl.COLORS[p] for p in fl.PATTERNS])


def _pattern_legend(ax_or_fig, loc="lower center", ncol=4, **kw):
    from matplotlib.patches import Patch

    handles = [Patch(fc=fl.COLORS[p], label=PATTERN_LABEL[p]) for p in fl.PATTERNS]
    return ax_or_fig.legend(handles=handles, loc=loc, ncol=ncol, fontsize=7, frameon=False, **kw)


# ------------------------------------------------------------------------------------------------ perfis estacionários
PROFILE_OPENINGS = (0.06, 0.13, 0.22, 0.36, 1.0)


def steady_profiles(out, png):
    """Estacionários a 15 bar para aberturas crescentes: fração de gás e padrão dominante ao longo do duto,
    pressão e velocidades das fases."""
    from matplotlib.patches import Patch

    from mfsim.analysis import PATTERN_FLAGS

    fig = plt.figure(figsize=(fl.DOUBLE, 0.66 * fl.DOUBLE))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.62, 1.0], hspace=0.42, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, :])
    ax_p = fig.add_subplot(gs[1, :], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_a)
    ax_v = fig.add_subplot(gs[2, 1], sharex=ax_a)
    data = [(ab, _steady_state(ab, P_SEL)) for ab in PROFILE_OPENINGS]
    data = [(ab, d) for ab, d in data if d is not None]
    if not data:
        for ax in (ax_a, ax_p, ax_c, ax_v):
            fl.placeholder(ax, "steady states of the map not found (tools/slug_map.py build)")
        fl.save(fig, out, "steady_profiles", png)
        return
    shades = plt.get_cmap("Blues")(np.linspace(0.45, 0.95, len(data)))
    patterns = []
    for (ab, (model, X)), c in zip(data, shades):
        pb = model.pipes[0]
        sch = pb.scheme
        N = sch.N
        Y = model.assemble(0.0, X)[1][0]
        s = np.asarray(sch.x_cells)
        Yc = Y[1 : N + 1].copy()
        Yc[:, 2] = 0.5 * (Y[0:N, 2] + Y[1 : N + 1, 2])
        Yc[:, 3] = 0.5 * (Y[0:N, 3] + Y[1 : N + 1, 3])
        mod = pb.model
        Yc[:, 0] = np.clip(Yc[:, 0], mod.params.alpha_min, mod.params.alpha_max)
        inp, _, _ = mod.local_state(Yc, sch.theta_nodes[1 : N + 1])
        _, flags, _ = mod.closures(inp)
        W = np.stack([np.asarray(getattr(flags, f), float) for _, f in PATTERN_FLAGS], axis=1)
        patterns.append((ab, sch.x_faces, W.argmax(1)))
        lab = f"$a = {ab:g}$"
        ax_a.plot(s, Y[1 : N + 1, 0], "-o", ms=2.0, lw=0.9, color=c, label=lab)
        ax_c.plot(s, Y[1 : N + 1, 1] * model.PR / 1e5, "-", lw=1.0, color=c)
        ax_v.plot(s, inp.vL, "-", lw=1.0, color=c)
        ax_v.plot(s, inp.vG, "--", lw=1.0, color=c)
    ax_a.set_ylabel(r"$\alpha_G$")
    ax_a.set_ylim(0, 1)
    ax_a.legend(ncol=len(data), fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    plt.setp(ax_a.get_xticklabels(), visible=False)
    fl.panel_label(ax_a, "(a)")
    for i, (ab, xf, dom) in enumerate(patterns):
        for j in range(dom.size):
            ax_p.add_patch(
                plt.Rectangle(
                    (xf[j], i - 0.42), xf[j + 1] - xf[j], 0.84, fc=fl.COLORS[fl.PATTERNS[dom[j]]], ec="white", lw=0.3
                )
            )
    ax_p.set_ylim(-0.6, len(patterns) - 0.4)
    ax_p.set_yticks(range(len(patterns)))
    ax_p.set_yticklabels([f"{ab:g}" for ab, *_ in patterns])
    ax_p.set_ylabel("opening")
    ax_p.grid(False)
    ax_p.legend(
        handles=[Patch(fc=fl.COLORS[p], label=PATTERN_LABEL[p]) for p in fl.PATTERNS],
        ncol=4,
        fontsize=6.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
    )
    plt.setp(ax_p.get_xticklabels(), visible=False)
    fl.panel_label(ax_p, "(b)")
    ax_c.set_ylabel("pressure (bar)")
    ax_c.set_xlabel("position along the pipe (m)")
    fl.panel_label(ax_c, "(c)")
    ax_v.set_ylabel(r"phase velocity (m s$^{-1}$)")
    ax_v.set_xlabel("position along the pipe (m)")
    ax_v.plot([], [], "-", color="0.35", label="liquid")
    ax_v.plot([], [], "--", color="0.35", label="gas")
    ax_v.legend(fontsize=6.5, loc="upper right")
    fl.panel_label(ax_v, "(d)")
    fl.save(fig, out, "steady_profiles", png)


# ------------------------------------------------------------------------------------------------ espaço–tempo
def _section_axis(ax, model, side="left"):
    """Eixo vertical por índice de célula (cada célula com a mesma altura, para a curva e o riser ficarem
    visíveis ao lado dos 300 m da linha), com as fronteiras dos trechos e os seus nomes."""
    z = _zones(model)
    th = z["th"]
    n = th.size
    k_line = int(np.argmax(th < -1.0))
    k_bend0 = int(z["k_bend"][0]) if z["k_bend"].size else n
    k_riser0 = int(z["k_riser"][0]) if z["k_riser"].size else n
    k_top0 = int(z["k_riser"][-1]) + 1 if z["k_riser"].size else n
    bounds = [
        (-0.5, k_line - 0.5, "inlet"),
        (k_line - 0.5, k_bend0 - 0.5, "line"),
        (k_bend0 - 0.5, k_riser0 - 0.5, "bend"),
        (k_riser0 - 0.5, k_top0 - 0.5, "riser"),
        (k_top0 - 0.5, n - 0.5, "top"),
    ]
    for lo, hi, lab in bounds:
        if hi <= lo:
            continue
        ax.axhline(lo, color="white", lw=0.6)
        if side == "right":
            ax.text(1.01, 0.5 * (lo + hi) / (n - 0.0) + 0.5 / n, lab, transform=ax.transAxes, fontsize=6, va="center")
    ax.set_ylim(-0.5, n - 0.5)
    ticks = [k_line, k_bend0, k_riser0, n - 1]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{z['s'][k]:.0f}" for k in ticks])
    return z


def _s_to_index(model, s_val):
    z = _zones(model)
    return np.interp(s_val, z["s"], np.arange(z["s"].size))


def spacetime(out, png):
    """Diagramas x–t da fração de líquido ao longo de todo o duto (eixo por célula) para três aberturas a
    15 bar, com as frentes (selo de líquido na linha e coluna no riser) e, por baixo, a pressão de entrada e a
    vazão de líquido à saída."""
    from mfsim.analysis import slug_fronts

    runs = [(ab, _dense_run(ab)) for ab in OPENINGS_SEL]
    fig = plt.figure(figsize=(fl.DOUBLE, 0.78 * fl.DOUBLE))
    gs = fig.add_gridspec(
        3, 5, height_ratios=[2.4, 0.75, 0.75], width_ratios=[1, 1, 1, 0.16, 0.045], hspace=0.12, wspace=0.14
    )
    rows = _map_rows() or []
    cls = {(r["start"], r["P_res"], r["ab"]): r["cls"] for r in rows}
    im, axes = None, np.empty((3, 3), dtype=object)
    for j, (ab, run) in enumerate(runs):
        ax0 = fig.add_subplot(gs[0, j])
        ax1 = fig.add_subplot(gs[1, j], sharex=ax0)
        ax2 = fig.add_subplot(gs[2, j], sharex=ax0)
        axes[:, j] = ax0, ax1, ax2
        if run is None:
            for a in (ax0, ax1, ax2):
                fl.placeholder(a, f"run a = {ab:g} not found")
            continue
        p = run.pipe(0)
        T = np.asarray(run.T)
        win = T >= T[-1] - 2400.0
        N = run.model.pipes[0].scheme.N
        hold = 1.0 - p["out"]["alphaG"][1 : N + 1][:, win]
        im = ax0.pcolormesh(
            T[win], np.arange(N), hold, cmap="Blues", vmin=0, vmax=1, shading="nearest", rasterized=True
        )
        _section_axis(ax0, run.model, side="right" if j == 2 else "left")
        fr = slug_fronts(run)
        z = _zones(run.model)
        k_low = float(_s_to_index(run.model, z["s_low"]))
        seal_idx = _s_to_index(run.model, z["s_low"] - fr["seal"][win])
        ax0.plot(T[win], seal_idx, color=fl.COLORS["annular"], lw=0.7)
        lev_s = (
            np.interp(fr["level"][win], [0.0, fr["H"]], [z["s_low"], z["s"][z["k_riser"][-1]]])
            if z["k_riser"].size
            else None
        )
        if lev_s is not None:
            ax0.plot(
                T[win],
                np.where(fr["level"][win] > 0, _s_to_index(run.model, lev_s), np.nan),
                color=fl.COLORS["intermittent"],
                lw=0.7,
            )
        lab = CLASS_LABEL.get(cls.get(("steady", P_SEL, ab)), "")
        ax0.set_title(f"$a = {ab:g}$" + (f"\n{lab}" if lab else ""), fontsize=7)
        ax1.plot(T[win], p["out"]["P_bar"][1][win], color=fl.INK, lw=0.8)
        ax2.plot(T[win], p["out"]["qL"][-1][win], color=fl.COLORS["stratified"], lw=0.8)
        ax2.set_xlabel("time (s)")
        for a in (ax0, ax1):
            plt.setp(a.get_xticklabels(), visible=False)
        if j > 0:
            for a in (ax1, ax2):
                plt.setp(a.get_yticklabels(), visible=False)
        fl.panel_label(ax0, f"({'abc'[j]})")
    axes[0, 0].set_ylabel("position along the pipe (m)")
    axes[1, 0].set_ylabel(r"$P_{in}$ (bar)")
    axes[2, 0].set_ylabel(r"$q_{L,out}$ (kg s$^{-1}$)")
    for i in (1, 2):
        lims = [a.get_ylim() for a in axes[i] if a is not None]
        for a in axes[i]:
            a.set_ylim(min(l[0] for l in lims), max(l[1] for l in lims))
    if im is not None:
        cax = fig.add_subplot(gs[0, 4])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("liquid holdup")
    fig.legend(
        handles=[
            plt.Line2D([], [], color=fl.COLORS["annular"], lw=1, label="end of the liquid seal in the line"),
            plt.Line2D([], [], color=fl.COLORS["intermittent"], lw=1, label="top of the liquid column in the riser"),
        ],
        loc="lower center",
        ncol=2,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.subplots_adjust(bottom=0.1)
    fl.save(fig, out, "spacetime", png)


# ------------------------------------------------------------------------------------------------ anatomia do ciclo
ANATOMY_OPENING = 0.6


def _cycle_events(T, P_in, fr):
    """Instantes que marcam as fases de um ciclo: riser cheio (a coluna chega ao topo), início do blowout (a
    coluna deixa de estar cheia: o gás entra pela base) e fim do blowout (mínimo da pressão de entrada)."""
    from mfsim.analysis import blowdowns

    H = fr["H"]
    full = np.flatnonzero((fr["level"][1:] >= 0.98 * H) & (fr["level"][:-1] < 0.98 * H)) + 1
    # início do blowout: a coluna de líquido no riser deixa de estar cheia (o gás entra pela base)
    open_ = np.flatnonzero((fr["level"][1:] < 0.98 * H) & (fr["level"][:-1] >= 0.98 * H)) + 1
    tb = blowdowns(T, P_in, min_gap=200.0) if np.ptp(P_in) > 0.3 else np.array([])
    return T[full], T[open_], tb


def cycle_anatomy(out, png):
    """Um ciclo de golfada severa: padrão dominante x–t em todo o duto (eixo por célula) com as frentes, as
    frentes em metros, as pressões na entrada e no ponto baixo e as vazões à saída; linhas verticais nos
    instantes em que o riser fica cheio, em que o gás chega ao ponto baixo e no fim do blowout."""
    from mfsim.analysis import cell_states, flow_pattern_weights, slug_fronts

    run = _dense_run(ANATOMY_OPENING)
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(fl.DOUBLE, 0.98 * fl.DOUBLE),
        sharex=True,
        gridspec_kw=dict(height_ratios=[2.2, 0.9, 0.8, 0.7, 0.7], hspace=0.16),
    )
    if run is None:
        for a in axes:
            fl.placeholder(a, "dense run not found (batches/paper01_dynamics.json)")
        fl.save(fig, out, "cycle_anatomy", png)
        return
    cs = cell_states(run)
    W = flow_pattern_weights(run, cells=cs)
    fr = slug_fronts(run, cells=cs)
    T = np.asarray(run.T)
    p = run.pipe(0)
    P_in = p["out"]["P_bar"][1]
    t_full, t_open, t_bd = _cycle_events(T, P_in, fr)
    # janela: dois blowouts completos no fim da corrida
    t1 = T[-1]
    t0 = t_bd[-3] - 100.0 if t_bd.size >= 3 else max(T[0], t1 - 1800.0)
    win = (T >= t0) & (T <= t1)
    dom = W["W"].argmax(-1)
    N = dom.shape[1]
    axes[0].pcolormesh(
        T[win], np.arange(N), dom[win].T, cmap=_pattern_cmap(), vmin=-0.5, vmax=3.5, shading="nearest", rasterized=True
    )
    _section_axis(axes[0], run.model, side="right")
    z = _zones(run.model)
    axes[0].plot(T[win], _s_to_index(run.model, z["s_low"] - fr["seal"][win]), color="white", lw=0.8)
    if z["k_riser"].size:
        lev_s = np.interp(fr["level"][win], [0.0, fr["H"]], [z["s_low"], z["s"][z["k_riser"][-1]]])
        axes[0].plot(
            T[win],
            np.where(fr["level"][win] > 0, _s_to_index(run.model, lev_s), np.nan),
            color="white",
            lw=0.8,
            ls=(0, (3, 1.5)),
        )
    axes[0].set_ylabel("position (m)")
    _pattern_legend(axes[0], loc="lower left", ncol=4, bbox_to_anchor=(0.0, 1.0))
    axes[1].plot(T[win], fr["seal"][win], color=fl.COLORS["stratified"], lw=1.0, label="liquid seal in the line")
    axes[1].plot(T[win], fr["level"][win], color=fl.COLORS["annular"], lw=1.0, label="liquid column in the riser")
    axes[1].set_ylabel("length (m)")
    kb = int(z["k_bend"][0]) + 1 if z["k_bend"].size else 1
    axes[2].plot(T[win], P_in[win], color=fl.INK, lw=0.9, label="line inlet")
    axes[2].plot(T[win], p["out"]["P_bar"][kb][win], color=fl.INK2, lw=0.9, ls="--", label="low point")
    axes[2].set_ylabel("P (bar)")
    axes[2].legend(fontsize=6.0, loc="lower right", ncol=2)
    axes[3].plot(T[win], p["out"]["qL"][-1][win], color=fl.COLORS["stratified"], lw=0.9)
    axes[3].set_ylabel(r"$q_L$ (kg s$^{-1}$)")
    axes[4].plot(T[win], 1e3 * p["out"]["qG"][-1][win], color=fl.COLORS["annular"], lw=0.9)
    axes[4].set_ylabel(r"$q_G$ (g s$^{-1}$)")
    axes[4].set_xlabel("time (s)")
    for a in axes[1:]:
        for tt, ls in ((t_full, (0, (1, 1.5))), (t_open, (0, (4, 2))), (t_bd, "-")):
            for t in tt[(tt >= t0) & (tt <= t1)]:
                a.axvline(t, color="0.6", lw=0.6, ls=ls, zorder=0)
    axes[1].plot([], [], color="0.6", lw=0.6, ls=(0, (1, 1.5)), label="riser full")
    axes[1].plot([], [], color="0.6", lw=0.6, ls=(0, (4, 2)), label="blowout starts")
    axes[1].plot([], [], color="0.6", lw=0.6, ls="-", label="end of blowout")
    axes[1].legend(fontsize=6.0, loc="upper left", ncol=5, bbox_to_anchor=(0.0, 1.32), frameon=False)
    for i, a in enumerate(axes):
        fl.panel_label(a, f"({'abcde'[i]})")
    fl.save(fig, out, "cycle_anatomy", png)


# ------------------------------------------------------------------------------------------------ mapa de estabilidade
def _jansen_line(rows):
    from mfsim.analysis import jansen_choke_dp_crit, jansen_choke_opening

    dpc = 0.5 * (jansen_choke_dp_crit(Phi=0.87) + jansen_choke_dp_crit())
    xs, ys = [], []
    for P in sorted({r["P_res"] for r in rows}):
        line = sorted((r["ab"], r["q_mean"]) for r in rows if r["P_res"] == P)
        a_c = jansen_choke_opening([t[0] for t in line], [t[1] for t in line], dpc, CV=15.0)
        if a_c is not None:
            xs.append(a_c)
            ys.append(P)
    return np.array(xs), np.array(ys), dpc


def _edges(v):
    v = np.log(np.asarray(v, float))
    mid = 0.5 * (v[1:] + v[:-1])
    return np.exp(np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]]))


def _draw_map(ax, rows, fixed_rows=None):
    from matplotlib.patches import Rectangle

    xs = sorted({r["ab"] for r in rows})
    ys = sorted({r["P_res"] for r in rows})
    xe, ye = _edges(xs), _edges(ys)
    G = {(r["P_res"], r["ab"]): r for r in rows}
    for i, P in enumerate(ys):
        best = None
        for j, ab in enumerate(xs):
            r = G.get((P, ab))
            if r is None:
                continue
            c = fl.COLORS.get(r["cls"] if r.get("success", True) else "", "#e1e0d9")
            ax.add_patch(
                Rectangle((xe[j], ye[i]), xe[j + 1] - xe[j], ye[i + 1] - ye[i], fc=c, ec="white", lw=0.8, alpha=0.85)
            )
            if r["cls"] == "stable":
                txt = f"{r['P_mean']:.1f}"
                if best is None or r["q_mean"] > best[0]:
                    best = (r["q_mean"], j)
            else:
                per = r.get("period", np.nan)
                txt = f"{r['P_max']:.1f}" + (
                    "" if not np.isfinite(per) else (f"\n{per / 60:.0f}m" if per >= 120 else f"\n{per:.0f}s")
                )
            ax.text(
                np.sqrt(xe[j] * xe[j + 1]),
                np.sqrt(ye[i] * ye[i + 1]),
                txt,
                ha="center",
                va="center",
                fontsize=4.6,
                color="white" if r["cls"] != "stable" else "white",
                linespacing=0.95,
            )
        if best is not None:
            j = best[1]
            ax.add_patch(
                Rectangle((xe[j], ye[i]), xe[j + 1] - xe[j], ye[i + 1] - ye[i], fc="none", ec=fl.INK, lw=1.4, zorder=5)
            )
    jx, jy, dpc = _jansen_line(rows)
    ax.plot(jx, jy, color=fl.INK, lw=1.2, ls=(0, (5, 3)), zorder=6)
    if fixed_rows:
        # fronteira do lado aberto com padrão fixo: primeira abertura instável depois da última estável em cada linha
        G2 = {(r["P_res"], r["ab"]): r["cls"] for r in fixed_rows}
        bx, by = [], []
        for i, P in enumerate(ys):
            stable = [j for j, ab in enumerate(xs) if G2.get((P, ab)) == "stable"]
            if stable:
                bx += [xe[max(stable) + 1]] * 2
                by += [ye[i], ye[i + 1]]
        ax.plot(bx, by, color=fl.INK2, lw=1.0, ls=(0, (1, 1.5)), zorder=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [t for t in (0.03, 0.05, 0.1, 0.2, 0.5, 1.0) if xe[0] <= t <= xe[-1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{v:g}" for v in ys])
    ax.minorticks_off()
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    ax.grid(False)
    return dpc


def stability_map(out, png):
    """Mapa abertura do choke × pressão do reservatório (padrão dinâmico) nas duas partidas, com o critério de
    Jansen et al. (1996) e a fronteira do lado aberto com padrão fixo."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows = _map_rows()
    fixed = _map_rows(MAP_FIXED_DIR)
    fig, axes = plt.subplots(1, 2, figsize=(fl.DOUBLE, 0.52 * fl.DOUBLE), sharey=True)
    dpc = None
    for ax, start, lab in zip(
        axes, ("steady", "blocked"), ("(a) start from the steady state", "(b) start from a blocked line")
    ):
        sub = [r for r in (rows or []) if r["start"] == start]
        if not sub:
            fl.placeholder(ax, "map not found (tools/slug_map.py)")
            continue
        dpc = _draw_map(ax, sub, [r for r in (fixed or []) if r["start"] == start])
        ax.set_xlabel("choke opening")
        ax.set_title(lab, fontsize=7.5, loc="left")
    axes[0].set_ylabel("reservoir pressure (bar)")
    handles = [Patch(fc=fl.COLORS[k], label=CLASS_LABEL[k]) for k in ("stable", "cycling", "slugging")]
    handles += [Patch(fc="none", ec=fl.INK, lw=1.4, label="most productive stable opening")]
    if dpc is not None:
        handles += [Line2D([], [], color=fl.INK, lw=1.2, ls=(0, (5, 3)), label=f"Jansen et al. (1996), {dpc:.2f} bar")]
    if fixed:
        handles += [
            Line2D([], [], color=fl.INK2, lw=1.0, ls=(0, (1, 1.5)), label="stable boundary, prescribed patterns")
        ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.24, wspace=0.05)
    fl.save(fig, out, "stability_map", png)


def bifurcation(out, png):
    """Diagrama de bifurcação a 15 bar: faixa da pressão de entrada em cada partida e o estacionário; produção
    média e a do estacionário."""
    rows = _map_rows()
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(fl.MID, 0.8 * fl.MID), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1.0], hspace=0.08)
    )
    if not rows:
        fl.placeholder(a1, "map not found")
        fl.placeholder(a2, "map not found")
        fl.save(fig, out, "bifurcation", png)
        return
    for start, lab in (("steady", "from the steady state"), ("blocked", "from a blocked line")):
        sub = sorted([r for r in rows if r["start"] == start and r["P_res"] == P_SEL], key=lambda r: r["ab"])
        ab = np.array([r["ab"] for r in sub])
        a1.fill_between(
            ab, [r["P_min"] for r in sub], [r["P_max"] for r in sub], color=fl.COLORS[start], alpha=0.2, lw=0
        )
        a1.plot(ab, [r["P_mean"] for r in sub], color=fl.COLORS[start], lw=1.2, label=f"{lab}: mean and range")
        a2.plot(ab, [r["q_mean"] for r in sub], "o-", ms=2.5, color=fl.COLORS[start], lw=1.2, label=f"{lab}")
    st = sorted([r for r in rows if r["start"] == "steady" and r["P_res"] == P_SEL], key=lambda r: r["ab"])
    a1.plot([r["ab"] for r in st], [r["P_ss"] for r in st], color=fl.INK, lw=1.0, ls=(0, (4, 3)), label="steady state")
    a2.plot([r["ab"] for r in st], [r["q_ss"] for r in st], color=fl.INK, lw=1.0, ls=(0, (4, 3)), label="steady state")
    jx, jy, _ = _jansen_line([r for r in rows if r["start"] == "steady"])
    a_c = jx[np.argmin(np.abs(jy - P_SEL))] if jx.size else None
    for a in (a1, a2):
        if a_c is not None:
            a.axvline(a_c, color="0.85", lw=5, zorder=0)
        a.set_xscale("log")
        a.legend(fontsize=7)
    a1.set_ylabel("inlet pressure (bar)")
    a2.set_ylabel(r"liquid production (kg s$^{-1}$)")
    a2.set_xlabel("choke opening")
    fl.panel_label(a1, "(a)")
    fl.panel_label(a2, "(b)")
    fl.save(fig, out, "bifurcation", png)


# ------------------------------------------------------------------------------------------------ tabelas dos lotes
def _batch_runs(batch):
    """Corridas de um lote do artigo (`runs/paper01/<lote>/<nome>.npz`), na ordem do ficheiro do lote, como
    lista de (nome, Run | None)."""
    from mfsim.runs import load_run

    spec = json.load(open(fl.PROJECT / "batches" / f"paper01_{batch}.json"))
    out = []
    for e in spec["runs"]:
        path = fl.RUNS / "paper01" / batch / f"{e['name']}.npz"
        out.append((e["name"], load_run(str(path)) if path.exists() else None))
    return out


def _run_summary(run):
    """Fim da corrida, sucesso, passos, CPU, erro de massa (líquido e gás, % do alimentado) e assinatura do
    ciclo (blowdowns pela pressão de entrada, intervalo médio, faixa de pressão, pico de líquido à saída)."""
    from mfsim.analysis import blowdowns, inventory_error

    info = run.info or {}
    T = np.asarray(run.T)
    p = run.pipe(0)
    P1, q = p["out"]["P_bar"][1], p["out"]["qL"][-1]
    sel = T > 300.0
    tb = blowdowns(T[sel], P1[sel]) if sel.sum() > 10 and np.ptp(P1[sel]) > 0.3 else np.array([])
    e = inventory_error(run)
    errL = errG = np.nan
    if e is not None:
        # erro no fim em % da massa que entrou pela fronteira de entrada (vazões da face de entrada)
        fedL = float(np.trapezoid(np.clip(p["out"]["qL"][0], 0.0, None), T))
        fedG = float(np.trapezoid(np.clip(p["out"]["qG"][0], 0.0, None), T))
        errL = 100.0 * float(e["L"][-1]) / fedL if fedL > 0 else np.nan
        errG = 100.0 * float(e["G"][-1]) / fedG if fedG > 0 else np.nan
    return dict(
        t_end=float(T[-1]),
        success=bool(info.get("success", True)),
        n_steps=info.get("n_steps"),
        cpu=info.get("cpu_total", info.get("cpu")),
        errL=errL,
        errG=errG,
        n_blowdowns=int(tb.size),
        interval=float(np.mean(np.diff(tb))) if tb.size > 1 else np.nan,
        P_min=float(P1[sel].min()) if sel.any() else np.nan,
        P_max=float(P1[sel].max()) if sel.any() else np.nan,
        q_peak=float(q[sel].max()) if sel.any() else np.nan,
    )


def _summary_rows(batch, labels):
    rows = []
    for name, run in _batch_runs(batch):
        lab = labels.get(name, name.replace("_", " "))
        if run is None:
            rows.append([lab] + ["--"] * 8)
            continue
        r = _run_summary(run)
        status = "completed" if r["success"] and r["t_end"] >= 2999 else f"stops at {r['t_end']:.0f} s"
        rows.append(
            [
                lab,
                status,
                fl.fmt(r["n_steps"], "{:,.0f}").replace(",", "\\,"),
                fl.fmt(r["cpu"], "{:.0f}"),
                fl.fmt(r["errL"], "{:+.1e}"),
                fl.fmt(r["errG"], "{:+.1e}"),
                str(r["n_blowdowns"]),
                fl.fmt(r["interval"], "{:.0f}"),
                f"{fl.fmt(r['P_min'])}--{fl.fmt(r['P_max'])}",
            ]
        )
    return rows


SUMMARY_HEADER = [
    "configuration",
    "outcome",
    "steps",
    "CPU (s)",
    "liquid error (\\%)",
    "gas error (\\%)",
    "blowouts",
    "interval (s)",
    "$P_{in}$ (bar)",
]
SUMMARY_NOTE = (
    "Mass errors: inventory change minus the boundary fluxes integrated with the integrator's own recurrence, "
    "at the end of the run, in per cent of the net feed. Blowouts: minima of the inlet pressure after \\SI{300}{\\second}."
)

DISC_LABELS = {
    "fv_conservative": "staggered FV, conservative",
    "fv_primitive": "staggered FV, primitive",
    "galerkin": "Galerkin",
    "petrov_galerkin": "Petrov--Galerkin",
    "fd_central": "central differences",
    "fd_upwind": "backward differences",
}
ABL_LABELS = {
    "reference": "configuration of Section 7 (reference)",
    "primitive": "primitive time integration",
    "relax_raw": "vanishing phase relaxed to rest",
    "no_flux_limiter": "no donor-cell flux limiter",
    "no_art_visc": "no acoustic viscosity",
    "closure_mean": "face closures with the mean void fraction",
    "ipe_thesis": "sharp, equilibrium-based pattern switch",
    "ipe_smooth_only": "smooth switch, equilibrium-based criteria",
    "no_low_void": "no low-void rule",
    "no_virtual_mass": "no virtual mass",
    "bubble_drag_thesis": "bubble drag without $g$ and $\\sigma/\\rho_L$",
    "bdf1": "first-order BDF",
    "uniform_atol": "single absolute tolerance $10^{-6}$",
    "newton12": "Newton budget of 12 iterations",
    "scipy_bdf": "general-purpose BDF (SciPy)",
    "gammaL1000": "liquid with $\\gamma_L = 1000$",
    "valve_thesis": "mass-weighted choke density",
    "sharp_bend": "sharp bend (no arc)",
}


def tab_discretisation(out):
    fl.tex_table(
        out,
        "tab_discretisation",
        SUMMARY_HEADER,
        _summary_rows("discretization", DISC_LABELS),
        align="llrrrrrrl",
        notes=SUMMARY_NOTE,
    )


def tab_ablation(out):
    fl.tex_table(
        out,
        "tab_ablation",
        SUMMARY_HEADER,
        _summary_rows("ablation", ABL_LABELS),
        align="llrrrrrrl",
        notes=SUMMARY_NOTE + " Minimum step \\SI{1e-6}{\\second}: a run that crawls stops with an error.",
    )


def discretisation(out, png):
    """Pressão de entrada com cada esquema até ao fim da corrida e erro de massa das duas formulações."""
    from mfsim.analysis import inventory_error

    fig, (a1, a2) = plt.subplots(
        2,
        1,
        figsize=(fl.DOUBLE, 0.55 * fl.DOUBLE),
        sharex=True,
        gridspec_kw=dict(height_ratios=[1.3, 1.0], hspace=0.08),
    )
    runs = dict(_batch_runs("discretization"))
    if not any(r is not None for r in runs.values()):
        fl.placeholder(a1, "runs not found (batches/paper01_discretization.json)")
        fl.placeholder(a2, "runs not found")
        fl.save(fig, out, "discretisation", png)
        return
    for name, lab in DISC_LABELS.items():
        r = runs.get(name)
        if r is None:
            continue
        key = name if name in fl.COLORS else ("fv_upwind" if name == "fv_conservative" else "primitive")
        T = np.asarray(r.T)
        a1.plot(T, r.pipe(0)["out"]["P_bar"][1], lw=0.9, color=fl.COLORS.get(key, fl.INK), label=lab)
        if not (r.info or {}).get("success", True) or T[-1] < 2999:
            a1.plot(T[-1], r.pipe(0)["out"]["P_bar"][1][-1], "x", ms=5, color=fl.COLORS.get(key, fl.INK))
    a1.set_ylabel(r"$P_{in}$ (bar)")
    a1.legend(fontsize=6.5, ncol=3, loc="upper right")
    fl.panel_label(a1, "(a)")
    for name, key, lab in (
        ("fv_conservative", "conservative", "conservative"),
        ("fv_primitive", "primitive", "primitive"),
    ):
        r = runs.get(name)
        if r is None:
            continue
        e = inventory_error(r)
        if e is None:
            continue
        a2.plot(e["T"], e["L"], color=fl.COLORS[key], lw=1.0, label=f"liquid, {lab}")
        a2.plot(e["T"], 100 * e["G"], color=fl.COLORS[key], lw=1.0, ls="--", label=f"gas (×100), {lab}")
    a2.axhline(0, color=fl.GRID, lw=0.6)
    a2.set_ylabel("inventory error (kg)")
    a2.set_xlabel("time (s)")
    a2.legend(fontsize=6.5, ncol=2)
    fl.panel_label(a2, "(b)")
    fl.save(fig, out, "discretisation", png)


# ------------------------------------------------------------------------------------------------ boa colocação na trajetória
TRAJ_RUN = ("dynamics", 0.6)  # corrida de golfada severa onde se mede a hiperbolicidade
TRAJ_VARIANTS = (
    ("model", {}),
    ("no_vm", dict(virtual_mass=0.0)),
    ("no_lv", dict(low_void_regime=0)),
    ("no_vm_no_lv", dict(virtual_mass=0.0, low_void_regime=0)),
    ("bestion", dict(interfacial_pressure=1.2)),
)


def _trajectory_compute():
    from mfsim.analysis import cell_states, characteristic_imag, flow_pattern_weights

    run = _dense_run(TRAJ_RUN[1])
    if run is None:
        return {"missing": np.array(1)}
    every = max(1, int(round(5.0 / max(np.median(np.diff(run.T)), 1e-9))))  # um estado a cada ~5 s
    cs = cell_states(run, every=every)
    W = flow_pattern_weights(run, cells=cs)
    out = {"dom": W["W"].argmax(-1), "theta": cs["theta"], "T": cs["T"]}
    for tag, kw in TRAJ_VARIANTS:
        out[f"imag_{tag}"] = characteristic_imag(run, cells=cs, **kw)
    # padrão dominante sem a regra de fração baixa (nos mesmos estados)
    return out


def tab_trajectory_wellposed(out):
    d = _cached("trajectory_wellposed", _trajectory_compute)
    names = {
        "model": "model (low-void rule + virtual mass)",
        "no_vm": "without virtual mass",
        "no_lv": "without low-void rule",
        "no_vm_no_lv": "without both",
        "bestion": "model + interfacial pressure",
    }
    if "missing" in d:
        rows = [[names[t]] + ["--"] * 6 for t, _ in TRAJ_VARIANTS]
    else:
        dom = d["dom"]
        th = np.rad2deg(d["theta"])
        rows = []
        for tag, _ in TRAJ_VARIANTS:
            bad = d[f"imag_{tag}"] > 1e-6
            row = [names[tag], fl.fmt(100 * bad.mean(), "{:.1f}")]
            for k in range(4):
                m = dom == k
                row.append(fl.fmt(100 * bad[m].mean(), "{:.1f}") if m.any() else "--")
            riser = np.broadcast_to(th > 85.0, bad.shape)
            row.append(fl.fmt(100 * bad[riser].mean(), "{:.1f}"))
            rows.append(row)
    fl.tex_table(
        out,
        "tab_trajectory_wellposed",
        ["closure set", "all (\\%)", "stratified", "intermittent", "bubbly", "annular", "riser"],
        rows,
        align="lrrrrrr",
        notes=(
            "Ill-posed fraction of the cell--time states (one state every \\SI{5}{\\second}) of the severe-slugging "
            f"run at an opening of {TRAJ_RUN[1]:g} and \\SI{{15}}{{\\bar}}, overall, by dominant flow pattern of the "
            "model and in the riser; the variants are evaluated a posteriori on the same states."
        ),
    )


# ------------------------------------------------------------------------------------------------ modos da curva
BEND_OPENINGS = (0.13, 0.28, 0.6)
ZONE_NAMES = ("inlet", "line", "front and bend", "riser")
ZONE_COLORS = {"inlet": "#E69F00", "line": "#0072B2", "front and bend": "#D55E00", "riser": "#009E73"}


def _zone_of_cells(model):
    """Zona (índice em ZONE_NAMES) de cada célula: entrada horizontal, linha descendente, frente e curva (última
    célula da linha e células da curva), riser e topo."""
    z = _zones(model)
    th = z["th"]
    zone = np.full(th.size, 1)
    zone[: int(np.argmax(th < -1.0))] = 0
    if z["k_bend"].size:
        zone[z["k_bend"][0] - 1 : z["k_bend"][-1] + 1] = 2
    zone[th > 89.0] = 3
    zone[np.flatnonzero(th > 89.0)[-1] + 1 :] = 3 if (th > 89.0).any() else zone[np.flatnonzero(th > 89.0)[-1] + 1 :]
    return zone


def _spectrum(model, X):
    from mfsim.compiled import get_jax

    _, jac = get_jax(model)
    J = np.asarray(jac(0.0, X))
    J = J.toarray() if hasattr(J, "toarray") else J
    w, V = np.linalg.eig(J)
    pb = model.pipes[0]
    nodes = pb.free // 4
    cell = np.clip(nodes[np.argmax(np.abs(V[pb.offset : pb.offset + pb.n_states]), axis=0)] - 1, 0, pb.scheme.N - 1)
    return w, _zone_of_cells(model)[cell]


def _bend_compute():
    from dataclasses import replace  # noqa: F401
    from mfsim.core.options import SchemeSpec
    from mfsim.io import case_from_json
    from mfsim.simulation import build_model, state_bounds
    from mfsim.steady import newton_steady

    out = {}
    for ab in BEND_OPENINGS:
        d = _steady_state(ab, P_SEL)
        if d is None:
            continue
        model, X = d
        w, zone = _spectrum(model, X)
        out[f"w_donor_{ab}"], out[f"z_donor_{ab}"] = w, zone
        out[f"a_donor_{ab}"] = model.assemble(0.0, X)[1][0][1 : model.pipes[0].scheme.N + 1, 0]
        # a mesma rede com os fechamentos das faces pela média: Newton a partir do estacionário doador
        net = case_from_json(str(MAP_DIR / "steady" / "cases" / f"{_name(ab, P_SEL)}.json"))
        pipe = net.pipes[0]
        opts = dict(pipe.scheme.options)
        opts["closure_alpha"] = "mean"
        pipe.scheme = SchemeSpec(type=pipe.scheme.type, options=tuple(sorted(opts.items())))
        m2 = build_model(net)
        X2, info = newton_steady(m2, X, tol=1e-9, verbose=False, bounds=state_bounds(m2), jax=True, max_iter=60)
        if info["converged"]:
            w2, z2 = _spectrum(m2, X2)
            out[f"w_mean_{ab}"], out[f"z_mean_{ab}"] = w2, z2
            out[f"a_mean_{ab}"] = m2.assemble(0.0, X2)[1][0][1 : m2.pipes[0].scheme.N + 1, 0]
    return out or {"missing": np.array(1)}


def bend_modes(out, png):
    """Estacionário e espetro da Jacobiana a 15 bar: (a) fração de gás junto à curva e no riser com os fechamentos
    das faces pela média e pela célula doadora (abertura 0,13); (b) espetro nas duas variantes na mesma abertura;
    (c, d) espetro com a doadora nas aberturas 0,28 e 0,6. Cor pela zona onde cada vetor próprio se concentra."""
    d = _cached("bend_modes", _bend_compute)
    fig, axes = plt.subplots(2, 2, figsize=(fl.DOUBLE, 0.62 * fl.DOUBLE), gridspec_kw=dict(hspace=0.35, wspace=0.22))
    ax_p, ax_b, ax_c, ax_d = axes.ravel()
    ab0 = BEND_OPENINGS[0]
    s_cells = None
    model_d = _steady_state(ab0, P_SEL)
    if model_d is not None:
        s_cells = np.asarray(model_d[0].pipes[0].scheme.x_cells)
    if f"a_donor_{ab0}" in d and s_cells is not None:
        sel = s_cells >= 260.0
        if f"a_mean_{ab0}" in d:
            ax_p.plot(
                s_cells[sel],
                d[f"a_mean_{ab0}"][sel],
                "o-",
                ms=2.5,
                lw=0.9,
                color=fl.COLORS["primitive"],
                label="mean face fraction",
            )
        ax_p.plot(
            s_cells[sel],
            d[f"a_donor_{ab0}"][sel],
            "s-",
            ms=2.5,
            lw=0.9,
            color=fl.COLORS["conservative"],
            label="donor face fraction",
        )
        z = _zones(model_d[0])
        for k, lab in (
            (z["k_bend"][0] if z["k_bend"].size else None, "bend"),
            (z["k_riser"][0] if z["k_riser"].size else None, "riser"),
        ):
            if k is not None:
                ax_p.axvline(s_cells[k] - 1.0, color=fl.GRID, lw=0.8)
                ax_p.text(s_cells[k] - 0.6, 0.86, lab, fontsize=6.5, color=fl.INK2)
        ax_p.set_xlabel("position along the pipe (m)")
        ax_p.set_ylabel(r"$\alpha_G$ at steady state")
        ax_p.legend(fontsize=6.5, loc="center right")
    else:
        fl.placeholder(ax_p, "steady state not available")
    fl.panel_label(ax_p, "(a)")

    def spectrum(ax, key, filled=True, label_zones=False):
        w, zone = d[key], d["z_" + key[2:]]
        for k, zn in enumerate(ZONE_NAMES):
            m = zone == k
            kw = (
                dict(s=7, color=ZONE_COLORS[zn], lw=0)
                if filled
                else dict(s=12, facecolors="none", edgecolors=ZONE_COLORS[zn], lw=0.6)
            )
            ax.scatter(w.real[m], w.imag[m], label=zn if label_zones else None, **kw)
        return w

    for ax, (ab, variants), lab in (
        (ax_b, (ab0, ("mean", "donor")), "(b)"),
        (ax_c, (BEND_OPENINGS[1], ("donor",)), "(c)"),
        (ax_d, (BEND_OPENINGS[2], ("donor",)), "(d)"),
    ):
        txt = []
        for var in variants:
            key = f"w_{var}_{ab}"
            if key not in d:
                continue
            w = spectrum(ax, key, filled=(var == "donor"), label_zones=(ax is ax_b and var == "donor"))
            txt.append(f"{var}: {int((w.real > 1e-9).sum())} unstable, max {w.real.max():.0f} s$^{{-1}}$")
        if not txt:
            fl.placeholder(ax, "steady state not available")
        else:
            ax.axvline(0, color=fl.INK, lw=0.6)
            ax.set_xlim(-30, 75)
            ax.set_ylim(-80, 80)
            ax.set_xlabel(r"Re $\lambda$ (s$^{-1}$)")
            ax.set_ylabel(r"Im $\lambda$ (rad s$^{-1}$)")
            ax.set_title(f"$a = {ab:g}$", fontsize=7.5)
            ax.text(
                0.98,
                0.04,
                "\n".join(txt),
                transform=ax.transAxes,
                ha="right",
                fontsize=6.0,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0),
            )
        fl.panel_label(ax, lab)
    ax_b.scatter([], [], s=12, facecolors="none", edgecolors="0.4", lw=0.6, label="mean (open)")
    ax_b.legend(fontsize=6.0, loc="upper left", ncol=1)
    fl.save(fig, out, "bend_modes", png)


# ------------------------------------------------------------------------------------------------ malha, desempenho e caso
MESH_LABELS = {
    "coarse": "coarse 15 + 5 (base)",
    "coarse_mean": "coarse, mean face fraction",
    "graded": "coarse, graded",
    "fine": "fine 30 + 10",
    "fine_mean": "fine, mean face fraction",
    "fine_graded": "fine, graded",
}


def tab_mesh(out):
    rows = []
    for name, run in _batch_runs("mesh"):
        lab = MESH_LABELS.get(name, name)
        if run is None:
            rows.append([lab] + ["--"] * 7)
            continue
        r = _run_summary(run)
        h = np.asarray(run.model.pipes[0].mesh["h"])
        rows.append(
            [
                lab,
                str(run.model.pipes[0].scheme.N),
                f"{h.min():.1f}--{h.max():.1f}",
                fl.fmt(r["n_steps"], "{:,.0f}").replace(",", "\\,"),
                fl.fmt(r["cpu"], "{:.0f}"),
                fl.fmt(r["interval"], "{:.0f}"),
                f"{fl.fmt(r['P_min'])}--{fl.fmt(r['P_max'])}",
                fl.fmt(r["q_peak"], "{:.1f}"),
            ]
        )
    fl.tex_table(
        out,
        "tab_mesh",
        ["mesh", "cells", "cell size (m)", "steps", "CPU (s)", "interval (s)", "$P_{in}$ (bar)", "peak $q_L$ (kg/s)"],
        rows,
        align="lrlrrrlr",
        notes="Choke open, reservoir at \\SI{15}{\\bar}, start from a blocked line, \\SI{4000}{\\second}.",
    )


def tab_performance(out):
    path = fl.RUNS / "paper01" / "performance" / "timings.json"
    t = json.load(open(path)) if path.exists() else {}
    runs = dict(_batch_runs("backend"))
    cpu = {k: ((r.info or {}).get("cpu_total") if r is not None else None) for k, r in runs.items()}
    steps = {k: ((r.info or {}).get("n_steps") if r is not None else None) for k, r in runs.items()}
    rows = [
        ["right-hand side (ms)", fl.fmt(t.get("numpy_rhs_ms"), "{:.2f}"), fl.fmt(t.get("jax_rhs_ms"), "{:.2f}")],
        [
            "Jacobian (ms)",
            fl.fmt(t.get("numpy_jac_ms"), "{:.1f}") + " (coloured FD)",
            fl.fmt(t.get("jax_jac_ms"), "{:.1f}") + " (exact)",
        ],
        [
            "compilation (s)",
            "--",
            fl.fmt((t.get("jax_compile_rhs_s") or np.nan) + (t.get("jax_compile_jac_s") or np.nan), "{:.0f}"),
        ],
        [
            "steps in \\SI{1000}{\\second}",
            fl.fmt(steps.get("numpy"), "{:,.0f}").replace(",", "\\,"),
            fl.fmt(steps.get("jax"), "{:,.0f}").replace(",", "\\,"),
        ],
        [
            "wall time for \\SI{1000}{\\second} (s)",
            fl.fmt(cpu.get("numpy"), "{:.0f}"),
            fl.fmt(cpu.get("jax"), "{:.0f}"),
        ],
    ]
    fl.tex_table(
        out,
        "tab_performance",
        ["", "NumPy", "JAX"],
        rows,
        align="lrr",
        notes=(
            f"Application case with the choke open; {t.get('n_states', '--')} states, "
            f"{t.get('n_colors', '--')} colours in the finite-difference Jacobian; parity of the two "
            f"right-hand sides {fl.fmt(t.get('parity_rhs'), '{:.0e}')}. One core of an Apple M5 Pro."
        ),
    )


def tab_case(out):
    net = _base_network()
    fl_ = net.fluid
    res = next(e for e in net.edges if type(e).__name__ == "Reservoir")
    ch = next(e for e in net.edges if type(e).__name__ == "Choke")
    pipe = net.pipes[0]
    integ = net.integration
    rows = [
        ["geometry", "horizontal 20 m, line 300 m at $-3^\\circ$, bend $R = 5$ m, riser 20 m, top 5 m; $D = 50$ mm"],
        ["mesh", f"{int(sum(pipe.n_elements))} cells (2 + 15 + 4 + 5 + 2)"],
        [
            "liquid",
            f"$\\rho_{{L0}} = {fl_.rhoL0:g}$ kg m$^{{-3}}$, $\\mu_L = {1e3 * fl_.muL:g}$ mPa s, $\\gamma_L = {fl_.gammaL:g}$",
        ],
        [
            "gas",
            f"$\\rho_{{G0}} = {fl_.rhoG0:g}$ kg m$^{{-3}}$ at {fl_.P0 / 1e5:g} bar, $\\mu_G = {1e6 * fl_.muG:g}$ $\\mu$Pa s, $\\gamma_G = {fl_.gammaG:g}$, $\\sigma = {fl_.sigma:g}$ N m$^{{-1}}$",
        ],
        [
            "inflow",
            f"$J_L = {res.IPL:g}$ kg s$^{{-1}}$ bar$^{{-1}}$, $J_G/J_L = {res.IPG / res.IPL:.2e}$, $P_{{res}}$ 8--48 bar",
        ],
        ["choke", f"homogeneous law, $c_v = {ch.CV:g}$, opening 0.03--1; separator 4 bar"],
        [
            "closures",
            "dynamic pattern (smooth, local, $k = 20$), low-void rule 1\\,\\%, $C_{vm} = 0.5$ ($\\lambda = 1$), physical bubble drag",
        ],
        [
            "scheme",
            "staggered FV, conservative, donor face closures, no-slip relaxation $\\tau = 0.05$ s, $\\nu = 0.05\\,c\\,h$",
        ],
        [
            "integrator",
            f"BDF 1--2, rtol {integ.rtol:g}, range-scaled atol {integ.tol_range:g}, Newton budget {integ.max_newton}, JAX",
        ],
    ]
    fl.tex_table(out, "tab_case", ["", ""], rows, align="lp{0.78\\linewidth}")


# ------------------------------------------------------------------------------------------------ tabela dos ciclos
DYN_OPENINGS = (0.13, 0.22, 0.28, 0.36, 0.6, 1.0)


def tab_cycle(out):
    """Assinatura dos ciclos a 15 bar nas corridas densas: classe, período, faixa de pressão, pico e média da
    produção, comprimento máximo do tampão (selo na linha mais coluna no riser), tempo de enchimento do riser
    e duração do blowout (médias sobre os ciclos depois do primeiro)."""
    from mfsim.analysis import cycle_class, slug_fronts

    rows = []
    for ab in DYN_OPENINGS:
        run = fl.load_run(f"paper01/dynamics/{_name(ab, P_SEL)}")
        if run is None:
            rows.append([f"{ab:g}"] + ["--"] * 8)
            continue
        T = np.asarray(run.T)
        p = run.pipe(0)
        P_in, q = p["out"]["P_bar"][1], p["out"]["qL"][-1]
        sel = T >= T[-1] - 3000.0
        c = cycle_class(T[sel], P_in[sel], q[sel])
        fr = slug_fronts(run)
        t_full, t_open, t_bd = _cycle_events(T, P_in, fr)
        fill = [t_full[t_full > tb][0] - tb for tb in t_bd if (t_full > tb).any()]
        blow = [tb - t_open[t_open < tb][-1] for tb in t_bd if (t_open < tb).any()]
        slug = fr["seal"][sel] + fr["level"][sel]
        rows.append(
            [
                f"{ab:g}",
                CLASS_LABEL[c["cls"]].split(",")[0],
                fl.fmt(c["period"], "{:.0f}"),
                f"{c['P_min']:.2f}--{c['P_max']:.2f}",
                fl.fmt(c["q_mean"], "{:.3f}"),
                fl.fmt(c["surge"], "{:.1f}"),
                fl.fmt(float(np.max(slug)) if c["cls"] != "stable" else np.nan, "{:.0f}"),
                fl.fmt(float(np.median(fill[1:] if len(fill) > 1 else fill)) if fill else np.nan, "{:.0f}"),
                fl.fmt(float(np.median(blow[1:] if len(blow) > 1 else blow)) if blow else np.nan, "{:.0f}"),
            ]
        )
    fl.tex_table(
        out,
        "tab_cycle",
        [
            "opening",
            "class",
            "period (s)",
            "$P_{in}$ (bar)",
            "$\\bar q_L$ (kg/s)",
            "peak/mean",
            "slug (m)",
            "riser fill (s)",
            "blowout (s)",
        ],
        rows,
        align="llrlrrrrr",
        notes=(
            "Reservoir at \\SI{15}{\\bar}, start from the perturbed steady state, last \\SI{3000}{\\second} of "
            "\\SI{6000}{\\second}. Slug: largest length of liquid that seals the system, in the line upstream of the "
            "low point plus the column in the riser. Riser fill: from the end of a blowout to a full riser; blowout: "
            "from the moment the column stops being full to the minimum of the inlet pressure (medians)."
        ),
    )


FIGURES = {
    "geometry": geometry,
    "hyperbolicity": hyperbolicity,
    "steady_profiles": steady_profiles,
    "spacetime": spacetime,
    "cycle_anatomy": cycle_anatomy,
    "stability_map": stability_map,
    "bifurcation": bifurcation,
    "discretisation": discretisation,
    "bend_modes": bend_modes,
}
TABLES = {
    "tab_hyperbolicity": tab_hyperbolicity,
    "tab_discretisation": tab_discretisation,
    "tab_ablation": tab_ablation,
    "tab_trajectory_wellposed": tab_trajectory_wellposed,
    "tab_mesh": tab_mesh,
    "tab_performance": tab_performance,
    "tab_case": tab_case,
    "tab_cycle": tab_cycle,
}

if __name__ == "__main__":
    fl.run_cli(FIGURES, __doc__.splitlines()[0], tables=TABLES)
