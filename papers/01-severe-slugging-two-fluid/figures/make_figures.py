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
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_common"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figlib as fl  # noqa: E402

PAPER = "paper01"
RUNS = fl.RUNS / PAPER
CACHE = RUNS / "cache"
BASE_CASE = fl.PROJECT / "cases" / "paper01" / "offshore.json"
DYN = os.environ.get("PAPER01_DYN", "paper01/offshore_dynamics")  # lote da dinâmica (aberturas do choke no ponto base)
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
    """Traçado do sistema offshore: poço, linha com o trecho descendente, curva em arco e riser, com a malha (centros
    das células) e os elementos de contorno (reservatório com IPR, choke, separador); detalhe da curva à escala."""
    from mfsim.geometry import pipe_mesh

    net = _base_network()
    pipes = {p.id: p for p in net.pipes}
    line, well = pipes["1"], pipes["well"]
    fig = plt.figure(figsize=(fl.DOUBLE, 0.42 * fl.DOUBLE))
    ax = fig.add_axes([0.08, 0.17, 0.6, 0.78])
    meshes = {}
    for pipe, c in ((well, fl.COLORS["bubble"]), (line, fl.COLORS["stratified"])):
        pts = np.asarray(pipe.points, float)
        mesh = pipe_mesh(pts[:, 0], pts[:, 1], pts[:, 2], np.asarray(pipe.n_elements, int), grading=pipe.grading)
        meshes[pipe.id] = mesh
        y, z = mesh["xyz"][:, 1], mesh["xyz"][:, 2]
        ax.plot(y, z, color=fl.INK, lw=1.4, solid_capstyle="round")
        ax.plot(0.5 * (y[1:] + y[:-1]), 0.5 * (z[1:] + z[:-1]), "o", ms=2.0, color=c)
    yw, zw = meshes["well"]["xyz"][:, 1], meshes["well"]["xyz"][:, 2]
    yl, zl = meshes["1"]["xyz"][:, 1], meshes["1"]["xyz"][:, 2]
    ax.annotate("reservoir (linear IPR)", xy=(yw[0], zw[0]), xytext=(yw[0] + 250, zw[0] + 60), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=fl.INK2))
    ax.annotate("choke, separator 10 bar", xy=(yl[-1], zl[-1]), xytext=(yl[-1] - 2600, zl[-1] - 120), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=fl.INK2))
    ax.text(2500, zl[0] + 60, "flowline, 6000 m", fontsize=7, color=fl.INK2, ha="center")
    ax.text(yl[-1] - 900, zl.min() + 30, "descent, 2°", fontsize=7, color=fl.INK2)
    ax.text(yl[-1] + 60, 0.5 * (zl.min() + zl.max()), "riser", fontsize=7, rotation=90, va="center")
    ax.text(yw[0] + 60, 0.5 * (zw[0] + zw[-1]), "well", fontsize=7, rotation=90, va="center")
    ax.axhline(0.0, color="0.7", lw=0.5, ls="--")
    ax.text(300, 25, "sea level", fontsize=6, color=fl.INK2)
    ax.set_xlabel("horizontal distance (m)")
    ax.set_ylabel("elevation (m)")
    ax.set_xlim(-200, yl[-1] + 300)
    ax.grid(True)
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", ls="", ms=3, color=fl.COLORS["bubble"], label="well cells"),
            plt.Line2D([], [], marker="o", ls="", ms=3, color=fl.COLORS["stratified"], label="flowline–riser cells"),
        ],
        loc="lower right",
        fontsize=7,
    )
    fl.panel_label(ax, "(a)")
    # detalhe da curva à escala
    ax2 = fig.add_axes([0.76, 0.17, 0.22, 0.78])
    mesh = meshes["1"]
    y, z = mesh["xyz"][:, 1], mesh["xyz"][:, 2]
    yc, zc = 0.5 * (y[1:] + y[:-1]), 0.5 * (z[1:] + z[:-1])
    sel = (y > y[-1] - 30) & (z < z.min() + 25)
    ax2.plot(y[sel], z[sel], color=fl.INK, lw=1.4)
    k = (yc > y[-1] - 30) & (zc < z.min() + 25)
    ax2.plot(yc[k], zc[k], "o", ms=3, color=fl.COLORS["stratified"])
    for yi, zi, th in zip(yc[k], zc[k], np.rad2deg(mesh["theta"])[k]):
        if 1 < th < 89:
            ax2.text(yi + 0.8, zi - 0.6, f"{th:.0f}°", fontsize=6, color=fl.INK2)
    ax2.set_aspect("equal")
    ax2.set_xlabel("horizontal distance (m)")
    ax2.set_title("bend, true scale", fontsize=7)
    ax2.grid(True)
    fl.panel_label(ax2, "(b)")
    fl.save(fig, out, "geometry", png)


# ------------------------------------------------------------------------------------------------ boa colocação
HYP_PRESSURE, HYP_VL = 20e5, 0.5  # Pa e m/s: estado típico da base do riser no caso base (offshore, 20 bar)
HYP_D = 0.1524  # m: diâmetro da linha e do riser
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
    D = HYP_D
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
            f"$v_L={HYP_VL:g}$ m/s, $P={HYP_PRESSURE / 1e5:g}$ bar, $D={1e3 * HYP_D:g}$ mm."
        ),
    )


# ------------------------------------------------------------------------------------------------ mapa e corridas
import os  # noqa: E402

MAP_DIR = Path(os.environ.get("PAPER01_MAP_DIR", str(fl.RUNS / "maps" / "well_auto")))
MAP_FIXED_DIR = fl.RUNS / "maps" / "well"  # padrão fixo (issue 043), para comparação
P_SEL = 200.0  # bar: pressão do reservatório do ponto base
SMOOTH = 20.0  # média móvel (s) da pressão de entrada na classe e no período (tools/slug_map.py --smooth)
OPENINGS_SEL = (1.0, 0.3, 0.1)  # choke aberto (golfada severa), intermédio, quase fechado (estabilizado)
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
    """Corrida do lote `paper01_offshore_dynamics` (abertura `ab` no ponto base, saída a 2 s) ou None."""
    return fl.load_run(f"{DYN}/ab{ab:.3f}")


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
        (-0.5, k_line - 0.5, "line"),
        (k_line - 0.5, k_bend0 - 0.5, "descent"),
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
        win = T >= T[-1] - SPACETIME_WINDOW
        N = run.model.pipes[0].scheme.N
        hold = 1.0 - p["out"]["alphaG"][1 : N + 1][:, win]
        im = ax0.pcolormesh(
            T[win], np.arange(N), hold, cmap="Blues", vmin=0, vmax=1, shading="nearest", rasterized=True
        )
        _section_axis(ax0, run.model, side="right" if j == 2 else "left")
        if j > 0:  # a escala de posições é a mesma nos três painéis: só o primeiro a mostra
            ax0.set_yticklabels([])
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
ANATOMY_OPENING = 1.0  # golfada severa com o choke aberto
SPACETIME_WINDOW = 8000.0  # s: janela dos diagramas espaço–tempo (toda a corrida)


def _cycle_events(T, P_in, fr, hold=20.0):
    """Fases de cada ciclo completo, alinhadas por ciclo k (entre dois fins de blowout): riser cheio (o nível chega
    a 0,98 da altura e fica lá pelo menos `hold` s), início do blowout (máximo da pressão de entrada com média móvel:
    o gás começa a entrar no riser) e fim do blowout (mínimo, `_blowdown_times` depois de 300 s). Devolve
    (t_full, t_peak, t_end, t_prev) com t_prev o fim do blowout anterior."""
    from mfsim.analysis import moving_average

    T = np.asarray(T, float)
    # pressão com média móvel de SMOOTH s: os picos de menos de um segundo (coluna do riser a oscilar contra o gás,
    # face da entrada) não marcam fases do ciclo
    P_in = moving_average(T, np.asarray(P_in, float), SMOOTH)
    late = T > T[0] + 300.0
    tb = _blowdown_times(T[late], P_in[late])
    full = np.asarray(fr["level"]) >= 0.98 * fr["H"]
    dt = float(np.median(np.diff(T)))
    n = max(1, int(np.ceil(hold / dt)))
    c = np.concatenate([[0], np.cumsum(full)])
    full_d = np.zeros(full.size, bool)
    full_d[: full.size - n + 1] = (c[n:] - c[:-n]) == n
    t_full, t_peak = [], []
    for k in range(1, tb.size):
        m = (T > tb[k - 1]) & (T < tb[k])
        t_peak.append(T[np.flatnonzero(m)[np.argmax(P_in[m])]])
        idx = np.flatnonzero(m & full_d)
        t_full.append(T[idx[0]] if idx.size else np.nan)
    return np.array(t_full), np.array(t_peak), tb[1:], tb[:-1]


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
    t_full, t_peak, t_end, t_prev = _cycle_events(T, P_in, fr)
    # janela: os dois últimos ciclos completos
    if t_end.size >= 2:
        t0, t1 = t_prev[-2] - 60.0, t_end[-1] + 120.0
    else:
        t0, t1 = max(T[0], T[-1] - 1500.0), T[-1]
    t1 = min(t1, T[-1])
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
    axes[2].legend(fontsize=6.0, loc="lower center", ncol=2)
    axes[3].plot(T[win], p["out"]["qL"][-1][win], color=fl.COLORS["stratified"], lw=0.9)
    axes[3].set_ylabel(r"$q_L$ (kg s$^{-1}$)")
    axes[4].plot(T[win], 1e3 * p["out"]["qG"][-1][win], color=fl.COLORS["annular"], lw=0.9)
    axes[4].set_ylabel(r"$q_G$ (g s$^{-1}$)")
    axes[4].set_xlabel("time (s)")
    for a in axes[1:]:
        for tt, ls in ((t_full, (0, (1, 1.5))), (t_peak, (0, (4, 2))), (t_end, "-")):
            for t in tt[np.isfinite(tt) & (tt >= t0) & (tt <= t1)]:
                a.axvline(t, color="0.6", lw=0.6, ls=ls, zorder=0)
    from matplotlib.lines import Line2D

    ev = [
        Line2D([], [], color="0.6", lw=0.8, ls=(0, (1, 1.5)), label="riser full"),
        Line2D([], [], color="0.6", lw=0.8, ls=(0, (4, 2)), label="blowout starts"),
        Line2D([], [], color="0.6", lw=0.8, ls="-", label="end of blowout"),
    ]
    fig.legend(handles=ev, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=6.5, frameon=False)
    fig.subplots_adjust(bottom=0.08)
    axes[1].legend(fontsize=6.0, loc="lower center", ncol=2, framealpha=0.9)
    axes[0].set_xlim(t0, t1)
    for i, a in enumerate(axes):
        fl.panel_label(a, f"({'abcde'[i]})")
    fl.save(fig, out, "cycle_anatomy", png)


CYCLE_PHASES = (
    ("end of blowout", 0.0),
    ("riser filling", 0.5),
    ("riser full", 1.0),
    ("production", 1.5),
    ("blowout starts", 2.0),
    ("blowout", 2.5),
)  # fração do intervalo entre eventos: 0 fim do blowout anterior, 1 riser cheio, 2 início do blowout, 3 fim


def cycle_profiles(out, png):
    """Perfis ao longo do duto em seis instantes do último ciclo completo da corrida da anatomia: fração de gás e
    padrão dominante junto à curva, pressão e vazão de líquido em todo o duto."""
    from matplotlib.patches import Patch

    from mfsim.analysis import cell_states, flow_pattern_weights, slug_fronts

    run = _dense_run(ANATOMY_OPENING)
    fig = plt.figure(figsize=(fl.DOUBLE, 0.7 * fl.DOUBLE))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.7, 1.0], hspace=0.62, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, :])
    ax_p = fig.add_subplot(gs[1, :], sharex=ax_a)
    ax_c, ax_q = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    if run is None:
        for ax in (ax_a, ax_p, ax_c, ax_q):
            fl.placeholder(ax, "dense run not found (batches/paper01_dynamics.json)")
        fl.save(fig, out, "cycle_profiles", png)
        return
    T = np.asarray(run.T)
    p = run.pipe(0)
    cs = cell_states(run)
    fr = slug_fronts(run, cells=cs)
    t_full, t_peak, t_end, t_prev = _cycle_events(T, p["out"]["P_bar"][1], fr)
    ok = np.flatnonzero(np.isfinite(t_full))
    if not ok.size:
        for ax in (ax_a, ax_p, ax_c, ax_q):
            fl.placeholder(ax, "no complete cycle in the run")
        fl.save(fig, out, "cycle_profiles", png)
        return
    k = ok[-1]
    ev = np.array([t_prev[k], t_full[k], t_peak[k], t_end[k]])
    times = [np.interp(f, [0, 1, 2, 3], ev) for _, f in CYCLE_PHASES]
    idx = [int(np.argmin(np.abs(T - t))) for t in times]
    W = flow_pattern_weights(run, cells=cs)
    dom = W["W"].argmax(-1)
    sch = run.model.pipes[0].scheme
    N = sch.N
    s_c, s_f = np.asarray(sch.x_cells), np.asarray(sch.x_faces)
    z = _zones(run.model)
    s0 = z["s_low"] - 60.0
    shades = plt.get_cmap("cividis")(np.linspace(0.0, 0.9, len(idx)))
    for (lab, _), i, c in zip(CYCLE_PHASES, idx, shades):
        a = p["out"]["alphaG"][1 : N + 1][:, i]
        ax_a.plot(s_c, a, "-o", ms=1.8, lw=0.9, color=c, label=f"{lab} ({T[i] - times[0]:.0f} s)")
        ax_c.plot(s_c, p["out"]["P_bar"][1 : N + 1][:, i], lw=1.0, color=c)
        ax_q.plot(s_f, p["out"]["qL"][0 : N + 1][:, i], lw=1.0, color=c)
    for j, i in enumerate(idx):
        k_dom = int(np.argmin(np.abs(cs["T"] - T[i])))
        for m in range(N):
            ax_p.add_patch(
                plt.Rectangle(
                    (s_f[m], j - 0.42),
                    s_f[m + 1] - s_f[m],
                    0.84,
                    fc=fl.COLORS[fl.PATTERNS[dom[k_dom, m]]],
                    ec="white",
                    lw=0.3,
                )
            )
    for ax in (ax_a, ax_p):
        for kk, lab in ((z["k_bend"], "bend"), (z["k_riser"], "riser")):
            if kk.size:
                ax.axvline(s_f[kk[0]], color=fl.INK, lw=0.6, ls=(0, (2, 2)))
    ax_a.set_xlim(s0, s_f[-1])
    ax_a.set_ylim(0, 1)
    ax_a.set_ylabel(r"$\alpha_G$")
    ax_a.legend(ncol=3, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, 1.32), frameon=False)
    plt.setp(ax_a.get_xticklabels(), visible=False)
    fl.panel_label(ax_a, "(a)")
    ax_p.set_ylim(len(idx) - 0.4, -0.6)
    ax_p.set_yticks(range(len(idx)))
    ax_p.set_yticklabels([lab for lab, _ in CYCLE_PHASES], fontsize=6)
    ax_p.grid(False)
    ax_p.legend(
        handles=[Patch(fc=fl.COLORS[q], label=PATTERN_LABEL[q]) for q in fl.PATTERNS],
        ncol=4,
        fontsize=6.3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=False,
    )
    fl.panel_label(ax_p, "(b)")
    ax_c.set_ylabel("pressure (bar)")
    ax_c.set_xlabel("position along the pipe (m)")
    fl.panel_label(ax_c, "(c)")
    ax_q.set_ylabel(r"$q_L$ (kg s$^{-1}$)")
    ax_q.set_xlabel("position along the pipe (m)")
    fl.panel_label(ax_q, "(d)")
    fl.save(fig, out, "cycle_profiles", png)


# ------------------------------------------------------------------------------------------------ mapa de estabilidade
def _dp_line(rows, dpc):
    """Abertura em que a queda no choke (lei monofásica do líquido, a produção de cada ponto) vale `dpc` bar, por
    pressão do reservatório."""
    from mfsim.analysis import jansen_choke_opening

    xs, ys = [], []
    for P in sorted({r["P_res"] for r in rows}):
        line = sorted((r["ab"], r["q_mean"]) for r in rows if r["P_res"] == P)
        a_c = jansen_choke_opening([t[0] for t in line], [t[1] for t in line], dpc, CV=15.0)
        if a_c is not None:
            xs.append(a_c)
            ys.append(P)
    return np.array(xs), np.array(ys)


def _jansen_line(rows):
    from mfsim.analysis import jansen_choke_dp_crit

    dpc = 0.5 * (jansen_choke_dp_crit(Phi=0.87) + jansen_choke_dp_crit())
    xs, ys = _dp_line(rows, dpc)
    return xs, ys, dpc


def _linear_dp():
    """Queda no choke na fronteira linear (cruzamento do modo global lento, interpolado em log da abertura) em
    cada pressão e a mediana entre 0,1 e 1 bar (exclui as linhas sem fronteira ligada, de pressão alta)."""
    d = _cached("map_spectrum", _map_spectrum_compute)
    if "rows" not in d:
        return {}, np.nan
    R = d["rows"]
    out = {}
    for P in np.unique(R[:, 1]):
        sub = R[R[:, 1] == P]
        sub = sub[np.argsort(sub[:, 0])]
        a_h = _crossing(sub[:, 0], sub[:, 2])
        if np.isfinite(a_h):
            q = np.exp(np.interp(np.log(a_h), np.log(sub[:, 0]), np.log(sub[:, 11])))
            out[float(P)] = (a_h, q**2 / 900.0 / (0.0075 * 15.0 * a_h) ** 2)
    dps = [dp for _, dp in out.values() if 0.1 < dp < 1.0]
    return out, float(np.median(dps)) if dps else np.nan


def _edges(v):
    v = np.log(np.asarray(v, float))
    mid = 0.5 * (v[1:] + v[:-1])
    return np.exp(np.concatenate([[v[0] - (mid[0] - v[0])], mid, [v[-1] + (v[-1] - mid[-1])]]))


def _linear_status():
    """Estado linear de cada célula do mapa (issue 048): {(pressão, abertura): (instável pelo modo global lento,
    modo rápido global da linha instável)}."""
    from mfsim.analysis import N_EFF_GLOBAL

    d = _cached("map_spectrum", _map_spectrum_compute)
    if "rows" not in d:
        return {}
    out = {}
    for r in d["rows"]:
        fast = bool(r[6] > 0 and r[8] >= N_EFF_GLOBAL and abs(r[7]) > 1.0)
        out[(float(r[1]), float(r[0]))] = (bool(r[2] > 0), fast)
    return out


def _draw_map(ax, rows, fixed_rows=None, linear=None, flag_fast=False, dp_fit=None):
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
    if dp_fit is not None and np.isfinite(dp_fit):
        fx, fy = _dp_line(rows, dp_fit)
        ax.plot(fx, fy, color=fl.INK, lw=1.0, ls=(0, (6, 2, 1, 2)), zorder=6)
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
    if linear:
        # fronteira linear do estacionário: contorno das células em que o modo global lento é instável
        def unstable(i, j):
            v = linear.get((ys[i], xs[j])) if 0 <= i < len(ys) and 0 <= j < len(xs) else None
            return None if v is None else v[0]

        for i in range(len(ys)):
            for j in range(len(xs)):
                u = unstable(i, j)
                if u is None:
                    continue
                for di, dj in ((0, 1), (1, 0)):
                    v = unstable(i + di, j + dj)
                    if v is None or v == u:
                        continue
                    if dj:
                        ax.plot([xe[j + 1]] * 2, [ye[i], ye[i + 1]], color=fl.INK, lw=1.6, zorder=7)
                    else:
                        ax.plot([xe[j], xe[j + 1]], [ye[i + 1]] * 2, color=fl.INK, lw=1.6, zorder=7)
        if flag_fast:
            for (P, ab), (_, fast) in linear.items():
                if fast and P in ys and ab in xs:
                    i, j = ys.index(P), xs.index(ab)
                    ax.plot(xe[j + 1] / 1.06, ye[i + 1] / 1.05, marker="*", ms=3.2, color="white", mec=fl.INK, mew=0.4)
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


LINEAR_LABEL = {
    "stable": "linearly stable",
    "cycling": "unstable slow pair (oscillatory)",
    "slugging": "unstable real global mode",
}


def _linear_rows():
    """Linhas do mapa linear (issue 048) no formato de `_map_rows`: classe pelo modo global lento dominante (estável,
    par complexo instável -> "cycling", real instável -> "slugging"), período do par, pressão e produção do
    estacionário."""
    d = _cached("map_spectrum", _map_spectrum_compute)
    if "rows" not in d:
        return []
    out = []
    for r in d["rows"]:
        ab, P, re1, im1 = r[0], r[1], r[2], r[3]
        cls = "stable" if not re1 > 0 else ("cycling" if im1 > 1e-9 else "slugging")
        out.append(
            dict(
                start="linear",
                ab=float(ab),
                P_res=float(P),
                cls=cls,
                period=2 * np.pi / im1 if cls == "cycling" else np.nan,
                P_mean=float(r[10]),
                P_max=float(r[10]),
                q_mean=float(r[11]),
                success=True,
            )
        )
    return out


def stability_map(out, png):
    """Mapa abertura do choke × pressão do reservatório (padrão dinâmico): (a) estabilidade linear do estacionário
    (issue 048); (b) classes das simulações a partir da linha bloqueada. Critério de Jansen et al. (1996), fronteira
    linear e fronteira do lado aberto com padrão fixo."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    lin_rows = _linear_rows()
    rows = [r for r in (_map_rows() or []) if r["start"] == "blocked"]
    fixed = [r for r in (_map_rows(MAP_FIXED_DIR) or []) if r["start"] == "blocked"]
    lin = _linear_status()
    _, dp_lin = _linear_dp()
    fig, axes = plt.subplots(1, 2, figsize=(fl.DOUBLE, 0.52 * fl.DOUBLE), sharey=True)
    dpc = None
    for ax, sub, fx, lab in (
        (axes[0], lin_rows, None, "(a) linear stability of the steady state"),
        (axes[1], rows, fixed, "(b) simulation from a blocked line"),
    ):
        if not sub:
            fl.placeholder(ax, "not available (tools/slug_map.py)")
            continue
        dpc = _draw_map(ax, sub, fx, linear=lin, flag_fast=ax is axes[0], dp_fit=dp_lin)
        ax.set_xlabel("choke opening")
        ax.set_title(lab, fontsize=7.5, loc="left")
    axes[0].set_ylabel("reservoir pressure (bar)")
    handles = [Patch(fc=fl.COLORS[k], label=f"{LINEAR_LABEL[k]} / {CLASS_LABEL[k]}") for k in CLASS_LABEL]
    handles += [Patch(fc="none", ec=fl.INK, lw=1.4, label="most productive stable opening")]
    handles += [Line2D([], [], color=fl.INK, lw=1.6, label="boundary of linear stability (slow modes)")]
    handles += [
        Line2D([], [], ls="none", marker="*", ms=5, color="white", mec=fl.INK, mew=0.5, label="fast line mode unstable")
    ]
    if dpc is not None:
        handles += [Line2D([], [], color=fl.INK, lw=1.2, ls=(0, (5, 3)), label=f"Jansen et al. (1996), {dpc:.2f} bar")]
    if np.isfinite(dp_lin):
        handles += [
            Line2D([], [], color=fl.INK, lw=1.0, ls=(0, (6, 2, 1, 2)), label=f"choke pressure drop of {dp_lin:.2f} bar")
        ]
    if fixed:
        handles += [
            Line2D([], [], color=fl.INK2, lw=1.0, ls=(0, (1, 1.5)), label="stable boundary, prescribed patterns")
        ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(bottom=0.28, wspace=0.05)
    fl.save(fig, out, "stability_map", png)


def bifurcation(out, png):
    """Diagrama de bifurcação a 15 bar: (a) pressão de entrada do estacionário (cheia onde é linearmente estável,
    tracejada onde não é) e faixa da pressão nas simulações a partir do estacionário perturbado (corridas densas)
    e da linha bloqueada (mapa); (b) produção média e a do estacionário. Linha vertical: abertura de Hopf."""
    from mfsim.analysis import cycle_class

    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(fl.MID, 0.8 * fl.MID), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1.0], hspace=0.08)
    )
    lin = sorted([r for r in _linear_rows() if r["P_res"] == P_SEL], key=lambda r: r["ab"])
    if not lin:
        fl.placeholder(a1, "steady states not found")
        fl.placeholder(a2, "steady states not found")
        fl.save(fig, out, "bifurcation", png)
        return
    ab = np.array([r["ab"] for r in lin])
    Pss = np.array([r["P_mean"] for r in lin])
    qss = np.array([r["q_mean"] for r in lin])
    st = np.array([r["cls"] == "stable" for r in lin])
    for a, y in ((a1, Pss), (a2, qss)):
        k = int(np.argmax(~st)) if (~st).any() else ab.size  # primeiro estacionário instável
        a.plot(ab, np.where(st, y, np.nan), color=fl.INK, lw=1.2, label="steady state, stable")
        a.plot(
            ab,
            np.where(~st | (np.arange(ab.size) == k - 1), y, np.nan),
            color=fl.INK,
            lw=1.0,
            ls=(0, (4, 3)),
            label="steady state, unstable",
        )
    # partida do estacionário: corridas densas
    dyn = []
    for a_run in DYN_OPENINGS:
        run = fl.load_run(f"paper01/dynamics/{_name(a_run, P_SEL)}")
        if run is None:
            continue
        T = np.asarray(run.T)
        pp = run.pipe(0)
        sel = T >= T[-1] - 3000.0
        c = cycle_class(T[sel], pp["out"]["P_bar"][1][sel], pp["out"]["qL"][-1][sel], smooth=SMOOTH)
        dyn.append((a_run, c))
    if dyn:
        x = np.array([d[0] for d in dyn])
        a1.vlines(x, [d[1]["P_min"] for d in dyn], [d[1]["P_max"] for d in dyn], color=fl.COLORS["steady"], lw=2.2)
        a1.plot(
            x,
            [d[1]["P_mean"] for d in dyn],
            "o",
            ms=3.5,
            color=fl.COLORS["steady"],
            label="from the steady state: mean and range",
        )
        a2.plot(x, [d[1]["q_mean"] for d in dyn], "o", ms=3.5, color=fl.COLORS["steady"], label="from the steady state")
    # partida bloqueada: mapa
    rows = sorted(
        [r for r in (_map_rows() or []) if r["start"] == "blocked" and r["P_res"] == P_SEL], key=lambda r: r["ab"]
    )
    if rows:
        xb = np.array([r["ab"] for r in rows])
        a1.fill_between(
            xb, [r["P_min"] for r in rows], [r["P_max"] for r in rows], color=fl.COLORS["blocked"], alpha=0.2, lw=0
        )
        a1.plot(
            xb,
            [r["P_mean"] for r in rows],
            color=fl.COLORS["blocked"],
            lw=1.1,
            label="from a blocked line: mean and range",
        )
        a2.plot(xb, [r["q_mean"] for r in rows], color=fl.COLORS["blocked"], lw=1.1, label="from a blocked line")
    h = _cached("hopf", _hopf_compute)
    if "base" in h:
        a_h = _crossing(h["base"][:, 0], h["base"][:, 1])
        for a in (a1, a2):
            a.axvline(a_h, color=fl.GRID, lw=0.8)
        a1.text(
            a_h * 1.04,
            0.97,
            f"$a_H = {a_h:.2f}$",
            fontsize=6.5,
            color=fl.INK2,
            va="top",
            transform=a1.get_xaxis_transform(),
        )
    for a in (a1, a2):
        a.set_xscale("log")
        a.legend(fontsize=6.2)
    a2.set_xticks([0.03, 0.1, 0.3, 1.0])
    a2.set_xticklabels(["0.03", "0.1", "0.3", "1"])
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

    path = fl.PROJECT / "batches" / f"paper01_{batch}.json"
    if not path.exists():
        return []
    spec = json.load(open(path))
    out = []
    for e in spec["runs"]:
        path = fl.RUNS / "paper01" / batch / f"{e['name']}.npz"
        out.append((e["name"], load_run(str(path)) if path.exists() else None))
    return out


def _blowdown_times(T, P, frac=0.5, min_gap=150.0):
    """Instantes dos blowdowns: mínimos da pressão de entrada abaixo de `frac` da faixa entre os percentis 2 e 98
    (robusta aos picos isolados), separados por pelo menos `min_gap` s. Vazio se a faixa for menor que 0,1 bar."""
    T, P = np.asarray(T, float), np.asarray(P, float)
    lo, hi = np.percentile(P, [2.0, 98.0])
    if hi - lo < 0.1:
        return np.array([])
    low = P < lo + frac * (hi - lo)
    out, i = [], 0
    while i < T.size:
        if low[i]:
            j = i
            while j < T.size and low[j]:
                j += 1
            k = i + int(np.argmin(P[i:j]))
            if not out or T[k] - out[-1] > min_gap:
                out.append(T[k])
            i = j
        else:
            i += 1
    return np.array(out)


def _run_summary(run):
    """Fim da corrida, sucesso, passos, CPU, erro de massa (líquido e gás, % do alimentado) e assinatura do
    ciclo (blowdowns pela pressão de entrada, intervalo médio, faixa de pressão, pico de líquido à saída)."""
    from mfsim.analysis import inventory_error

    info = run.info or {}
    T = np.asarray(run.T)
    p = run.pipe(0)
    P1, q = p["out"]["P_bar"][1], p["out"]["qL"][-1]
    sel = T > 300.0
    tb = _blowdown_times(T[sel], P1[sel]) if sel.sum() > 10 else np.array([])
    e = inventory_error(run)
    errL = errG = np.nan
    if e is not None:
        # erro no fim em % da massa que entrou pela fronteira de entrada (vazões da face de entrada)
        fedL = float(np.trapezoid(np.clip(p["out"]["qL"][0], 0.0, None), T))
        fedG = float(np.trapezoid(np.clip(p["out"]["qG"][0], 0.0, None), T))
        errL = 100.0 * float(e["L"][-1]) / fedL if fedL > 0 else np.nan
        errG = 100.0 * float(e["G"][-1]) / fedG if fedG > 0 else np.nan
    import re

    t_end = float(T[-1])
    m = re.search(r"em t=([0-9.eE+-]+)", str(info.get("error") or ""))
    if m:  # falha entre instantes de saída: o instante do erro é mais exato do que a última saída
        t_end = max(t_end, float(m.group(1)))
    return dict(
        t_end=t_end,
        stopped_by=("budget" if "orçamento" in str(info.get("error") or "") else ("min_step" if m else None)),
        success=bool(info.get("success", True)),
        n_steps=info.get("n_steps"),
        cpu=info.get("cpu_total", info.get("cpu")),
        errL=errL,
        errG=errG,
        n_blowdowns=int(tb.size),
        interval=float(np.mean(np.diff(tb))) if tb.size >= 2 else np.nan,
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
        if r["success"] and r["t_end"] >= 2999:
            status = "completed"
        elif r["stopped_by"] == "budget":
            status = f"too slow, at {r['t_end']:.0f} s" if r["t_end"] >= 100 else f"too slow, at {r['t_end']:.3g} s"
        else:
            status = f"fails at {r['t_end']:.3g} s"
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
                f"{fl.fmt(r['P_min'])}--{fl.fmt(r['P_max'])}" if np.isfinite(r["P_min"]) else "--",
            ]
        )
    return rows


SUMMARY_HEADER = [
    "configuration",
    "outcome",
    "steps",
    "wall time (s)",
    "liquid error (\\%)",
    "gas error (\\%)",
    "blowdowns",
    "interval (s)",
    "$P_{in}$ (bar)",
]
SUMMARY_NOTE = (
    "Mass errors: inventory change minus the boundary fluxes integrated with the integrator's own recurrence, at "
    "the end of the run, in per cent of the mass fed. Blowdowns: minima of the inlet pressure below the middle of "
    "its range (2nd to 98th percentile) after \\SI{300}{\\second}; interval: mean time between them. Wall-clock "
    "times with eight to twelve simulations sharing the machine; a run is stopped after \\SI{5400}{\\second}, 1.8 "
    "times the time of the reference."
)

DISC_LABELS = {
    "fv_conservative": "staggered FV, conservative",
    "fv_primitive": "staggered FV, primitive",
    "galerkin": "Galerkin",
    "petrov_galerkin": "Petrov--Galerkin",
    "fd_central": "central differences",
    "fd_upwind": "backward differences",
}
DISC_SHORT = {
    "fv_conservative": "FV, conservative",
    "fv_primitive": "FV, primitive",
    "galerkin": "Galerkin",
    "petrov_galerkin": "Petrov--Galerkin",
    "fd_central": "central diff.",
    "fd_upwind": "backward diff.",
}
ABL_LABELS = {
    "reference": "reference (Section~\\ref{sec:application})",
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
    "base": "reference (Section~\\ref{sec:application})",
    "unit_consistent": "slug unit of the thesis, mass-consistent",
    "unit_thesis": "slug unit of the thesis",
    "freq_thesis": "slug frequency of the thesis",
    "k_switch_80": "logistic switches with $k = 80$",
}


def tab_discretisation(out):
    fl.tex_table(
        out,
        "tab_discretisation",
        SUMMARY_HEADER,
        _summary_rows("offshore_discretization", DISC_LABELS),
        align="llrrrrrrl",
        notes=SUMMARY_NOTE,
    )


def tab_ablation(out):
    fl.tex_table(
        out,
        "tab_ablation",
        SUMMARY_HEADER,
        _summary_rows("offshore_ablation", ABL_LABELS),
        align="llrrrrrrl",
        notes=SUMMARY_NOTE,
    )


def discretisation(out, png):
    """(a) Pressão de entrada nos primeiros 100 s com os seis esquemas (cruz: onde a corrida falha); (b) a corrida
    inteira com os volumes finitos conservativos e primitivos; (c) erro dos estoques de líquido e de gás das duas
    formulações (acumulador BDF)."""
    from mfsim.analysis import inventory_error

    fig = plt.figure(figsize=(fl.DOUBLE, 0.62 * fl.DOUBLE))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 2.0], hspace=0.12, wspace=0.2)
    a1 = fig.add_subplot(gs[:, 0])
    a2 = fig.add_subplot(gs[0, 1])
    a3 = fig.add_subplot(gs[1, 1], sharex=a2)
    runs = dict(_batch_runs("discretization"))
    if not any(r is not None for r in runs.values()):
        for a in (a1, a2, a3):
            fl.placeholder(a, "runs not found (batches/paper01_discretization.json)")
        fl.save(fig, out, "discretisation", png)
        return
    # as duas formulações do esquema escalonado têm as cores de conservativo e primitivo; o Galerkin, que na paleta
    # dos esquemas partilha o vermelhão do primitivo, fica cinzento nesta figura
    colors = {
        "fv_conservative": fl.COLORS["conservative"],
        "fv_primitive": fl.COLORS["primitive"],
        "galerkin": fl.INK2,
    }
    for name, lab in DISC_LABELS.items():
        r = runs.get(name)
        if r is None:
            continue
        c = colors.get(name, fl.COLORS.get(name, fl.INK))
        T = np.asarray(r.T)
        P = r.pipe(0)["out"]["P_bar"][1]
        sel = T <= 100.0
        summ = _run_summary(r)
        fails = summ["t_end"] < 2999
        verb = "too slow, stopped at" if summ["stopped_by"] == "budget" else "fails at"
        short = DISC_SHORT[name] + (f", {verb} {summ['t_end']:.2g} s" if fails else "")
        a1.plot(T[sel], P[sel], lw=0.9, color=c, label=short)
        if fails:
            a1.plot([min(summ["t_end"], 100.0)], [P[-1]], "x", ms=6, mew=1.2, color=c)
        if name in ("fv_conservative", "fv_primitive"):
            a2.plot(T, P, lw=0.7, color=c, label=DISC_SHORT[name])
    a1.set_xlim(-2, 100)
    a1.set_xlabel("time (s)")
    a1.set_ylabel(r"$P_{in}$ (bar)")
    a1.legend(fontsize=6, loc="upper right", handlelength=1.2)
    fl.panel_label(a1, "(a)")
    a2.set_ylabel(r"$P_{in}$ (bar)")
    a2.legend(fontsize=6.5, ncol=2, loc="lower right")
    plt.setp(a2.get_xticklabels(), visible=False)
    fl.panel_label(a2, "(b)")
    err_cons = 0.0
    for name, key in (("fv_conservative", "conservative"), ("fv_primitive", "primitive")):
        r = runs.get(name)
        e = inventory_error(r) if r is not None else None
        if e is None:
            continue
        a3.plot(e["T"], e["L"], color=fl.COLORS[key], lw=1.0)
        a3.plot(e["T"], 100 * e["G"], color=fl.COLORS[key], lw=1.0, ls=(0, (4, 2)))
        if key == "conservative":
            err_cons = max(np.abs(e["L"]).max(), np.abs(e["G"]).max())
    a3.axhline(0, color=fl.GRID, lw=0.6)
    a3.set_ylabel("mass error (kg)")
    a3.set_xlabel("time (s)")
    # cor = formulação (legenda de (b)); estilo = fase
    a3.legend(
        handles=[
            plt.Line2D([], [], color=fl.INK, lw=1.0, label="liquid"),
            plt.Line2D([], [], color=fl.INK, lw=1.0, ls=(0, (4, 2)), label=r"gas $\times 100$"),
        ],
        fontsize=6.5,
        ncol=2,
        loc="upper left",
    )
    if err_cons > 0:
        ex = int(np.floor(np.log10(err_cons)))
        m = int(np.ceil(err_cons / 10.0**ex))
        a3.annotate(
            rf"conservative: |error| $<{m}\times10^{{{ex}}}$ kg",
            (0.99, 0.52),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=6.5,
            color=fl.INK,
        )
    fl.panel_label(a3, "(c)")
    fl.save(fig, out, "discretisation", png)


# ------------------------------------------------------------------------------------------------ boa colocação na trajetória
TRAJ_RUN = ("dynamics", 1.0)  # corrida de golfada severa onde se mede a hiperbolicidade
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
    """Autovalores da linearização M·δẋ = J·δx (com a matriz de massa do modo conservativo, issue 048) e a zona
    da célula onde cada vetor próprio se concentra."""
    from mfsim.analysis import linear_spectrum

    w, V = linear_spectrum(model, X, jax=True, vectors=True)
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
    "base": "base (15 + 57 cells)",
    "coarse": "coarse (8 + 32 cells)",
    "fine": "fine (30 + 112 cells)",
}


def tab_mesh(out):
    rows = []
    for name, run in [("base", _dense_run(1.0))] + _batch_runs("offshore_mesh"):
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
                f"{fl.fmt(r['P_min'])}--{fl.fmt(r['P_max'])}" if np.isfinite(r["P_min"]) else "--",
                fl.fmt(r["q_peak"], "{:.1f}"),
            ]
        )
    fl.tex_table(
        out,
        "tab_mesh",
        ["mesh", "cells", "cell size (m)", "steps", "CPU (s)", "interval (s)", "$P_{in}$ (bar)", "peak $q_L$ (kg/s)"],
        rows,
        align="lrlrrrlr",
        notes="Choke open at the base point, blocked start; cells of the flowline--riser pipe; base run from the openings batch.",
    )


def tab_performance(out):
    path = fl.RUNS / "paper01" / "performance" / "timings.json"
    loaded = not path.exists()  # na falta da medida com a máquina livre, a medida sob carga
    if loaded:
        path = path.with_name("timings_loaded.json")
    t = json.load(open(path)) if path.exists() else {}
    runs = dict(_batch_runs("offshore_backend"))
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
            "steps in \\SI{2000}{\\second}",
            fl.fmt(steps.get("numpy"), "{:,.0f}").replace(",", "\\,"),
            fl.fmt(steps.get("jax"), "{:,.0f}").replace(",", "\\,"),
        ],
        [
            "wall time for \\SI{2000}{\\second} (s)",
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
            f"right-hand sides {fl.fmt(t.get('parity_rhs'), '{:.0e}')}. Apple M5 Pro"
            + (" \\tbd{(measured with other simulations running)}." if loaded else ".")
        ),
    )


def tab_case(out):
    net = _base_network()
    d = _base_case_dict()
    fl_ = net.fluid
    res = next(e for e in net.edges if type(e).__name__ == "Reservoir")
    ch = next(e for e in net.edges if type(e).__name__ == "Choke")
    pipes = {p.id: p for p in net.pipes}
    integ = net.integration
    mk, ms = d["meta"]["market"], d["meta"]["mass"]
    rows = [
        [
            "geometry",
            "well \\SI{1500}{\\metre} vertical ($D = \\SI{127}{\\milli\\metre}$); flowline \\SI{5000}{\\metre} horizontal and "
            "\\SI{1000}{\\metre} at \\SI{-2}{\\degree}, bend of radius \\SI{5}{\\metre}, riser \\SI{334.9}{\\metre}, top "
            "\\SI{10}{\\metre} ($D = \\SI{152.4}{\\milli\\metre}$)",
        ],
        ["mesh", f"{int(sum(pipes['well'].n_elements))} cells in the well, {int(sum(pipes['1'].n_elements))} in the flowline--riser (25 + 10 + 4 + 16 + 2)"],
        [
            "liquid",
            f"$\\rho_{{L0}} = \\SI{{{fl_.rhoL0:g}}}{{\\kilogram\\per\\cubic\\metre}}$ at \\SI{{{fl_.P0 / 1e5:g}}}{{\\bar}}, "
            f"$\\mu_L = \\SI{{{1e3 * fl_.muL:g}}}{{\\milli\\pascal\\second}}$, $\\gamma_L = {fl_.gammaL:g}$",
        ],
        [
            "gas",
            f"$\\rho_{{G0}} = \\SI{{{fl_.rhoG0:g}}}{{\\kilogram\\per\\cubic\\metre}}$ at \\SI{{{fl_.P0 / 1e5:g}}}{{\\bar}}, "
            f"$\\mu_G = \\SI{{{1e6 * fl_.muG:g}}}{{\\micro\\pascal\\second}}$, $\\gamma_G = {fl_.gammaG:g}$, "
            f"$\\sigma = \\SI{{{fl_.sigma:g}}}{{\\newton\\per\\metre}}$",
        ],
        [
            "well",
            f"$J = \\SI{{{mk['PI_Sm3_d_bar']:g}}}{{\\cubic\\metre\\per\\day\\per\\bar}}$, $P_{{res}} = \\SI{{{mk['P_res_bar']:g}}}{{\\bar}}$, "
            f"$\\mathrm{{GOR}} = \\SI{{{mk['GOR_Sm3_Sm3']:g}}}{{\\cubic\\metre\\per\\cubic\\metre}}$ ($w_G = {ms['w_gas']:.3f}$; "
            f"$J_L = \\SI{{{res.IPL:.3f}}}{{\\kilogram\\per\\second\\per\\bar}}$, $J_G = \\SI{{{res.IPG:.4f}}}{{\\kilogram\\per\\second\\per\\bar}}$)",
        ],
        ["choke", f"homogeneous law, $c_v = {ch.CV:g}$, opening 0.1 to 1; separator at \\SI{{10}}{{\\bar}}"],
        [
            "closures",
            "dynamic flow pattern (smooth, local, $k = 20$), low-void rule below \\SI{1}{\\percent}, "
            "$C_{vm} = 0.5$ ($\\lambda = 1$), physical bubble drag, slug unit with the film at equilibrium",
        ],
        [
            "scheme",
            "staggered finite volumes, conservative, donor face closures, no-slip relaxation with "
            "$\\tau = \\SI{0.05}{\\second}$, $\\nu = 0.05\\,c\\,h$",
        ],
        [
            "integrator",
            f"BDF of order 1--2, rtol {integ.rtol:g}, range-scaled atol {integ.tol_range:g}, Newton budget "
            f"{integ.max_newton}, JAX",
        ],
        ["start", "blocked: flowline stratified at equilibrium, bend and riser full of liquid"],
    ]
    fl.tex_table(out, "tab_case", ["component", "setting"], rows, align="lp{0.78\\linewidth}")


# ------------------------------------------------------------------------------------------------ razão gás–óleo
GOR_RUNS = ((30.0, "gor30"), (50.0, "gor50"), (80.0, "gor80"), (122.13, "gor122.13"))
GOR_BASE = 122.13


def _gor_run(g, name):
    """Corrida do lote `paper01_offshore_gor` (ou a do ponto base, que é a mesma do lote da dinâmica)."""
    run = fl.load_run(f"paper01/offshore_gor/{name}")
    return run if run is not None else (_dense_run(1.0) if abs(g - GOR_BASE) < 1e-9 else None)


def gor(out, png):
    """Mais e menos gás do poço com o choke aberto: pressão de entrada da linha e vazão de líquido à saída para
    quatro razões gás–óleo, e a fração de líquido do riser."""
    from mfsim.analysis import slug_fronts

    fig, axes = plt.subplots(3, 1, figsize=(fl.DOUBLE, 0.62 * fl.DOUBLE), sharex=True, gridspec_kw=dict(hspace=0.12))
    shades = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(GOR_RUNS)))
    any_run = False
    for (g, name), c in zip(GOR_RUNS, shades):
        run = _gor_run(g, name)
        if run is None:
            continue
        any_run = True
        T = np.asarray(run.T)
        p = run.pipe(0)
        fr = slug_fronts(run)
        lab = f"GOR = {g:g} m$^3$/m$^3$"
        axes[0].plot(T, p["out"]["P_bar"][1], color=c, lw=0.8, label=lab)
        axes[1].plot(T, p["out"]["qL"][-1], color=c, lw=0.8)
        axes[2].plot(T, fr["level"] / fr["H"], color=c, lw=0.8)
    if not any_run:
        for a in axes:
            fl.placeholder(a, "runs of the gas--oil ratio batch not found")
    axes[0].set_ylabel(r"$P_{in}$ (bar)")
    axes[1].set_ylabel(r"$q_{L,out}$ (kg s$^{-1}$)")
    axes[2].set_ylabel("riser liquid column / height")
    axes[2].set_xlabel("time (s)")
    axes[0].legend(fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.25), frameon=False)
    for i, a in enumerate(axes):
        fl.panel_label(a, f"({'abc'[i]})")
    fl.save(fig, out, "gor", png)


def tab_gor(out):
    """Assinatura do ciclo por razão gás–óleo (choke aberto): fração de gás, vazão média, classe, intervalo entre
    blowdowns, faixa de pressão e pico de líquido."""
    rows = []
    for g, name in GOR_RUNS:
        run = _gor_run(g, name)
        if run is None:
            rows.append([f"{g:g}", "--", "--", "--", "--", "--", "--"])
            continue
        r = _run_summary(run)
        w = (run.case.meta or {}).get("mass", {}).get("w_gas", np.nan) if getattr(run, "case", None) is not None else np.nan
        rows.append(
            [
                f"{g:g}",
                fl.fmt(w, "{:.3f}"),
                fl.fmt(r.get("q_mean"), "{:.2f}"),
                r.get("cls", "--") if isinstance(r.get("cls"), str) else "--",
                fl.fmt(r["interval"], "{:.0f}"),
                f"{fl.fmt(r['P_min'])}--{fl.fmt(r['P_max'])}" if np.isfinite(r["P_min"]) else "--",
                fl.fmt(r["q_peak"], "{:.1f}"),
            ]
        )
    fl.tex_table(
        out,
        "tab_gor",
        ["GOR (m$^3$/m$^3$)", "$w_G$", "mean $q_L$ (kg/s)", "class", "interval (s)", "$P_{in}$ (bar)", "peak $q_L$ (kg/s)"],
        rows,
        align="rrrlrlr",
        notes="Choke open at the base point, blocked start, \\SI{8000}{\\second}; signatures over the last \\SI{4000}{\\second}.",
    )


# ------------------------------------------------------------------------------------------------ tabela dos ciclos
DYN_OPENINGS = (1.0, 0.6, 0.3, 0.2, 0.1)


def tab_cycle(out):
    """Assinatura dos ciclos a 15 bar nas corridas densas (últimos 3000 s): classe, período, faixa de pressão,
    produção média e razão pico/média, comprimento máximo do selo de líquido na linha, fração de líquido mínima do
    riser (distingue a oscilação, em que o riser nunca esvazia, da golfada de relaxação), queda da pressão de
    entrada (do máximo ao mínimo) e reenchimento do riser (só nos ciclos em que ele esvazia; medianas)."""
    from mfsim.analysis import cycle_class, slug_fronts

    rows = []
    for ab in DYN_OPENINGS:
        run = fl.load_run(f"paper01/dynamics/{_name(ab, P_SEL)}")
        if run is None:
            rows.append([f"{ab:g}"] + ["--"] * 9)
            continue
        T = np.asarray(run.T)
        p = run.pipe(0)
        P_in, q = p["out"]["P_bar"][1], p["out"]["qL"][-1]
        sel = T >= T[-1] - 3000.0
        c = cycle_class(T[sel], P_in[sel], q[sel], smooth=SMOOTH)
        fr = slug_fronts(run)
        t_full, t_peak, t_end, t_prev = _cycle_events(T, P_in, fr)
        late = t_end >= T[-1] - 3000.0
        hold = np.asarray(fr["riser_holdup"])
        fall = [b - a for a, b, k in zip(t_peak, t_end, late) if k]
        refill = []
        for f, a, b, k in zip(t_full, t_prev, t_end, late):
            m = (T >= a) & (T <= b)
            if k and np.isfinite(f) and m.any() and hold[m].min() < 0.5:
                refill.append(f - a)
        cyc = c["cls"] != "stable"
        rows.append(
            [
                f"{ab:g}",
                CLASS_LABEL[c["cls"]].split(",")[0],
                fl.fmt(c["period"], "{:.0f}"),
                f"{c['P_min']:.2f}--{c['P_max']:.2f}",
                fl.fmt(c["q_mean"], "{:.3f}"),
                fl.fmt(c["surge"], "{:.1f}"),
                fl.fmt(float(np.max(fr["seal"][sel])) if cyc else np.nan, "{:.0f}"),
                fl.fmt(float(np.min(hold[sel])), "{:.2f}"),
                fl.fmt(float(np.median(fall)) if cyc and fall else np.nan, "{:.0f}"),
                fl.fmt(float(np.median(refill)) if refill else np.nan, "{:.0f}"),
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
            "seal (m)",
            "min.\\ riser holdup",
            "fall of $P_{in}$ (s)",
            "riser refill (s)",
        ],
        rows,
        align="llrlrrrrrr",
        notes=(
            "Reservoir at \\SI{15}{\\bar}, start from the perturbed steady state, last \\SI{3000}{\\second} of "
            "\\SI{6000}{\\second}. Seal: largest length of liquid-filled line upstream of the low point. Fall of "
            "$P_{in}$: from its maximum to its minimum in a cycle; riser refill: from the end of a blowout to a full "
            "riser (level at \\SI{98}{\\percent} of its height for at least \\SI{20}{\\second}), only in the cycles in "
            "which the mean liquid holdup of the riser falls below 0.5 (medians over the cycles)."
        ),
    )


# ------------------------------------------------------------------------------------------------ estabilidade linear
HOPF_OPENINGS = tuple(float(a) for a in np.round(np.r_[np.arange(0.10, 0.40, 0.01), np.arange(0.40, 1.0001, 0.05)], 3))
HOPF_MESHES = (
    ("base", None, "coarse 15 + 5 (base)"),
    ("fine", "stable_fine", "fine 30 + 10"),
    ("fine_graded", "stable_fine_graded", "fine, graded"),
)  # semente a 0,13: o estacionário do mapa (base) ou Newton a partir do fim da corrida estável do lote de malha


def _modes(model, X):
    """Autovalores (Im >= 0), nós efetivos e nó de maior peso (`mfsim.analysis.mode_participation`, issue 048)."""
    from mfsim.analysis import mode_participation

    return mode_participation(model, X, jax=True)


def _mode_row(w, n_eff, node):
    """[Re1, Im1, Re2, Im2, Re_o, Im_o, n_eff_o, nó_o] de `mfsim.analysis.slow_global_modes`: os dois modos globais
    lentos de maior parte real e, entre os outros (rápidos ou locais), o de maior parte real."""
    from mfsim.analysis import slow_global_modes

    g = slow_global_modes(w, n_eff, node)
    return [g[k] for k in ("re1", "im1", "re2", "im2", "re_other", "im_other", "n_eff_other", "node_other")]


def _map_spectrum_compute():
    """Modos do estacionário de cada ponto do mapa com padrão dinâmico (casos da partida do estacionário):
    linhas [abertura, pressão] + `_mode_row` + [pressão de entrada (bar), vazão de líquido à saída (kg/s)]."""
    from mfsim.io import case_from_json
    from mfsim.simulation import build_model

    rows = []
    for path in sorted((Path(MAP_DIR) / "steady" / "cases").glob("*.json")):
        net = case_from_json(str(path))
        if net.initial != "steady":
            continue
        m = (json.load(open(path)).get("meta") or {}).get("map") or {}
        model = build_model(net)
        X = model.initial_state()
        Y = model.assemble(0.0, X)[1][0]
        N = model.pipes[0].scheme.N
        rows.append([m["ab"], m["P_res"]] + _mode_row(*_modes(model, X)) + [Y[1, 1] * model.PR / 1e5, Y[N, 2]])
    return {"rows": np.array(rows)} if rows else {"missing": np.array(1)}


def _hopf_compute():
    """Continuação em abertura a 15 bar (Newton a partir do estacionário vizinho) nas três malhas: linhas
    [abertura] + `_mode_row` + [N] por malha."""
    import copy

    from mfsim.io import case_from_json, load_json
    from mfsim.runs import load_run
    from mfsim.simulation import build_model, state_bounds
    from mfsim.steady import newton_steady

    out = {}
    for key, run_name, _ in HOPF_MESHES:
        if run_name is None:
            seed = _steady_state(0.13, P_SEL)
            path = Path(MAP_DIR) / "steady" / "cases" / f"{_name(0.13, P_SEL)}.json"
            if seed is None:
                continue
            src, X0 = load_json(str(path)), seed[1]
        else:
            path = fl.RUNS / "paper01" / "mesh" / f"{run_name}.npz"
            if not path.exists():
                continue
            run = load_run(str(path))
            src, X0 = run.case_json, np.asarray(run.X[:, -1], float)
        res = {}
        for up in (True, False):
            X = X0
            seq = [a for a in HOPF_OPENINGS if (a >= 0.13 - 1e-9) == up]
            for ab in seq if up else seq[::-1]:
                d = copy.deepcopy(src)
                model = build_model(case_from_json(d, overrides={"network.edges.choke_2.opening": ab}))
                X, info = newton_steady(
                    model, X, tol=1e-10, verbose=False, bounds=state_bounds(model), jax=True, max_iter=60
                )
                if not info["converged"]:
                    break
                res[ab] = [ab] + _mode_row(*_modes(model, X)) + [model.pipes[0].scheme.N]
        if res:
            out[key] = np.array([res[a] for a in sorted(res)])
    return out or {"missing": np.array(1)}


def _crossing(x, y):
    """Primeiro x em que y passa de negativo a positivo (interpolação linear em log x) ou NaN."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    k = np.flatnonzero((y[:-1] < 0) & (y[1:] >= 0))
    if not k.size:
        return np.nan
    k = k[0]
    lx = np.log(x[k]) + (np.log(x[k + 1]) - np.log(x[k])) * (-y[k]) / (y[k + 1] - y[k])
    return float(np.exp(lx))


def tab_hopf_mesh(out):
    """Abertura crítica (o par lento atravessa o eixo imaginário) e período nesse ponto a 15 bar, contra a malha."""
    d = _cached("hopf", _hopf_compute)
    rows = []
    for key, _, lab in HOPF_MESHES:
        if key not in d:
            rows.append([lab] + ["--"] * 4)
            continue
        A = d[key]
        a_h = _crossing(A[:, 0], A[:, 1])
        im_h = np.interp(np.log(a_h), np.log(A[:, 0]), A[:, 2]) if np.isfinite(a_h) else np.nan
        k36 = int(np.argmin(np.abs(A[:, 0] - 0.36)))
        rows.append(
            [
                lab,
                f"{int(A[0, -1])}",
                fl.fmt(a_h, "{:.3f}"),
                fl.fmt(2 * np.pi / im_h if im_h > 0 else np.nan, "{:.0f}"),
                fl.fmt(1e3 * A[k36, 1], "{:+.1f}"),
                fl.fmt(2 * np.pi / A[k36, 2] if A[k36, 2] > 0 else np.nan, "{:.0f}"),
            ]
        )
    fl.tex_table(
        out,
        "tab_hopf_mesh",
        [
            "mesh",
            "cells",
            "critical opening",
            "period at onset (s)",
            "growth at 0.36 ($10^{-3}$ s$^{-1}$)",
            "period at 0.36 (s)",
        ],
        rows,
        align="lrrrrr",
        notes=(
            "Slow pair of the linearised steady state at a reservoir pressure of \\SI{15}{\\bar}: the opening at which "
            "it crosses the imaginary axis (interpolated) and its period there, and its growth rate and period at the "
            "opening of 0.36."
        ),
    )


FACE_COLORS = {"donor": fl.INK, "mean": "#CC79A7"}  # fração das faces nos fechamentos (044)


def _dyn_periods():
    """Classe e período (média móvel de SMOOTH s, últimos 3000 s) das corridas densas a 15 bar: {abertura: (classe,
    período)}."""
    from mfsim.analysis import cycle_class

    out = {}
    for ab in DYN_OPENINGS:
        run = fl.load_run(f"paper01/dynamics/{_name(ab, P_SEL)}")
        if run is None:
            continue
        T = np.asarray(run.T)
        p = run.pipe(0)
        sel = T >= T[-1] - 3000.0
        c = cycle_class(T[sel], p["out"]["P_bar"][1][sel], p["out"]["qL"][-1][sel], smooth=SMOOTH)
        out[ab] = (c["cls"], c["period"])
    return out


def linear_stability(out, png):
    """Estabilidade linear do estacionário (matriz de massa, issue 048) a 15 bar: (a) fração de gás junto à curva e
    no riser com os fechamentos das faces pela média e pela doadora (abertura 0,13); (b) os modos lentos nas duas
    variantes; (c) o lugar do par lento ao abrir o choke; (d) a sua taxa de crescimento e a do modo global real;
    (e) o seu período e o das corridas."""
    from matplotlib.lines import Line2D

    d = _cached("bend_modes", _bend_compute)
    h = _cached("hopf", _hopf_compute)
    fig = plt.figure(figsize=(fl.DOUBLE, 0.72 * fl.DOUBLE))
    gs = fig.add_gridspec(2, 6, hspace=0.42, wspace=1.1)
    ax_p, ax_s = fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6])
    ax_l, ax_g, ax_t = fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6])
    ab0 = BEND_OPENINGS[0]
    # (a) perfil junto à curva
    md = _steady_state(ab0, P_SEL)
    if md is not None and f"a_donor_{ab0}" in d:
        s_cells = np.asarray(md[0].pipes[0].scheme.x_cells)
        sel = s_cells >= 260.0
        for var, mk in (("mean", "o"), ("donor", "s")):
            if f"a_{var}_{ab0}" in d:
                ax_p.plot(
                    s_cells[sel],
                    d[f"a_{var}_{ab0}"][sel],
                    mk + "-",
                    ms=2.6,
                    lw=0.9,
                    color=FACE_COLORS[var],
                    label=f"{var} face fraction",
                )
        z = _zones(md[0])
        for k, lab in ((z["k_bend"], "bend"), (z["k_riser"], "riser")):
            if k.size:
                ax_p.axvline(s_cells[k[0]] - 1.0, color=fl.GRID, lw=0.8)
                ax_p.text(s_cells[k[0]] - 0.4, 0.84, lab, fontsize=6.5, color=fl.INK2)
        ax_p.set_xlabel("position along the pipe (m)")
        ax_p.set_ylabel(r"$\alpha_G$ at steady state")
        ax_p.legend(fontsize=6.5, loc="center right")
    else:
        fl.placeholder(ax_p, "steady state not available")
    fl.panel_label(ax_p, "(a)")
    # (b) modos lentos e intermédios a 0,13, média contra doadora
    for var, kw in (
        ("mean", dict(s=16, facecolors="none", edgecolors=FACE_COLORS["mean"], lw=0.7)),
        ("donor", dict(s=8, color=FACE_COLORS["donor"], lw=0)),
    ):
        key = f"w_{var}_{ab0}"
        if key in d:
            w = d[key]
            w = w[(w.imag >= 0) & (w.real > -0.06) & (w.imag < 0.32)]
            ax_s.scatter(w.real, w.imag, label=f"{var} face fraction", **kw)
            un = w[w.real > 1e-9]
            for lam in un:
                ax_s.annotate(
                    f"{2 * np.pi / lam.imag:.0f} s" if lam.imag > 1e-9 else "real",
                    (lam.real, lam.imag),
                    textcoords="offset points",
                    xytext=(5, -2),
                    fontsize=6.0,
                    color=fl.INK2,
                )
    ax_s.axvline(0, color=fl.INK, lw=0.6)
    ax_s.set_xlim(-0.06, 0.025)
    ax_s.set_ylim(0, 0.32)
    ax_s.set_xlabel(r"Re $\lambda$ (s$^{-1}$)")
    ax_s.set_ylabel(r"Im $\lambda$ (rad s$^{-1}$)")
    ax_s.legend(fontsize=6.5, loc="upper left")
    ax_s.set_title(f"$a = {ab0:g}$", fontsize=7.5)
    fl.panel_label(ax_s, "(b)")
    # (c-e) continuação em abertura
    A = h.get("base")
    if A is None:
        for ax in (ax_l, ax_g, ax_t):
            fl.placeholder(ax, "continuation not available")
    else:
        ab = A[:, 0]
        cplx = A[:, 2] > 1e-9
        re_p = np.where(cplx, A[:, 1], A[:, 3])  # o par complexo (1.º ou 2.º modo global)
        im_p = np.where(cplx, A[:, 2], A[:, 4])
        re_r = np.where(cplx, np.nan, A[:, 1])  # o modo global real, quando domina
        a_h = _crossing(ab, re_p)
        ax_l.plot(re_p, im_p, "-", color=fl.INK, lw=1.0)
        for a_lab in (0.13, 0.22, 0.28, 0.36, 0.6, 1.0):
            k = int(np.argmin(np.abs(ab - a_lab)))
            ax_l.plot(re_p[k], im_p[k], "o", ms=3, color=fl.INK)
            ax_l.annotate(f"{a_lab:g}", (re_p[k], im_p[k]), textcoords="offset points", xytext=(4, -6), fontsize=6)
        ax_l.axvline(0, color=fl.INK, lw=0.6)
        ax_l.set_xlabel(r"Re $\lambda$ ($10^{-3}$ s$^{-1}$)")
        ax_l.set_ylabel(r"Im $\lambda$ (rad s$^{-1}$)")
        ax_l.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{1e3 * v:g}"))
        fl.panel_label(ax_l, "(c)")
        # (d) crescimento contra a abertura, nas malhas
        styles = {"base": dict(color=fl.INK, lw=1.1), "fine": dict(color=fl.INK2, lw=0.9, ls=(0, (4, 2)))}
        styles["fine_graded"] = dict(color=fl.INK2, lw=0.9, ls=(0, (1, 1.5)))
        handles = []
        for key, _, lab in HOPF_MESHES:
            B = h.get(key)
            if B is None:
                continue
            c2 = B[:, 2] > 1e-9
            ax_g.plot(B[:, 0], 1e3 * np.where(c2, B[:, 1], B[:, 3]), **styles[key])
            handles.append(Line2D([], [], label=lab.replace(" 15 + 5", "").replace(" 30 + 10", ""), **styles[key]))
        ax_g.axhline(0, color=fl.INK, lw=0.6)
        ax_g.set_ylim(-7, 32)
        if np.isfinite(re_r).any():
            # o modo global real (relaxação) cresce uma ordem de grandeza mais depressa: fora da escala, indicado
            a_r = ab[np.isfinite(re_r)]
            ax_g.annotate(
                "",
                xy=(a_r[0], 31.5),
                xytext=(a_r[0], 22),
                arrowprops=dict(arrowstyle="->", color=fl.COLORS["slugging"], lw=1.1),
            )
            ax_g.text(
                a_r[0] * 1.04,
                31,
                "real\nglobal\nmode",
                ha="left",
                va="top",
                fontsize=5.8,
                color=fl.INK2,
                bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85),
            )
        if np.isfinite(a_h):
            ax_g.axvline(a_h, color=fl.GRID, lw=0.8)
            ax_g.text(a_h * 1.05, -6, f"$a_H = {a_h:.2f}$", fontsize=6.5, color=fl.INK2, va="bottom")
        ax_g.set_xscale("log")
        ax_g.set_xticks([0.1, 0.2, 0.5, 1.0])
        ax_g.set_xticklabels(["0.1", "0.2", "0.5", "1"])
        ax_g.set_xlabel("choke opening")
        ax_g.set_ylabel(r"growth rate ($10^{-3}$ s$^{-1}$)")
        ax_g.legend(handles=handles, fontsize=5.8, loc="upper left", handlelength=1.6)
        fl.panel_label(ax_g, "(d)")
        # (e) período do par contra o das corridas
        ax_t.plot(ab, 2 * np.pi / im_p, color=fl.INK, lw=1.1, label="slow pair")
        dyn = _dyn_periods()
        for a_run, (cls, per) in dyn.items():
            if cls != "stable" and np.isfinite(per):
                ax_t.plot(a_run, per, "o", ms=4.5, color=fl.COLORS[cls], zorder=5)
        for cls in ("cycling", "slugging"):
            if any(c == cls for c, _ in dyn.values()):
                ax_t.plot(
                    [], [], "o", ms=4.5, color=fl.COLORS[cls], label=f"simulated, {CLASS_LABEL[cls].split(',')[0]}"
                )
        if np.isfinite(a_h):
            ax_t.axvline(a_h, color=fl.GRID, lw=0.8)
        ax_t.set_xscale("log")
        ax_t.set_xticks([0.1, 0.2, 0.5, 1.0])
        ax_t.set_xticklabels(["0.1", "0.2", "0.5", "1"])
        ax_t.set_xlabel("choke opening")
        ax_t.set_ylabel("period (s)")
        ax_t.legend(fontsize=5.8, loc="upper right")
        fl.panel_label(ax_t, "(e)")
    fl.save(fig, out, "linear_stability", png)


# ------------------------------------------------------------------------------------------------ problema-modelo
def _toy():
    """Módulo do problema-modelo (`toy/model_problem.py`, linha de transmissão + terminação de resistência
    negativa) e os seus parâmetros."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "model_problem", Path(__file__).resolve().parent.parent / "toy" / "model_problem.py"
    )
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    return mp, mp.params()


TOY_SLOW = 4.4  # módulo máximo dos modos lentos do problema-modelo (o primeiro acústico da linha tem |λ| ≈ 4,55)


def _toy_scan():
    """Os dois modos lentos de maior parte real (|λ| < TOY_SLOW, Im >= 0) com a matriz de massa e o autovalor
    dominante de J sozinha, contra a resistência `a`."""
    mp, p = _toy()
    A = np.array(mp.A_SCAN)
    W = []
    for a in A:
        w = mp.spectrum(a, p)
        w = w[(np.abs(w) < TOY_SLOW) & (w.imag >= -1e-12)]
        W.append(np.r_[w[:2], np.full(max(0, 2 - w.size), np.nan + 0j)])
    WJ = np.array([mp.spectrum(a, p, with_mass=False)[0] for a in A])
    return A, np.array(W), WJ


def model_problem(out, png):
    """Problema-modelo: (a) plano de fase da terminação, característica f(y) + a y e ciclos simulados; (b) parte
    real dos modos lentos contra a resistência a, com a matriz de massa, e do autovalor dominante de J sozinha;
    (c) período do par e dos ciclos simulados; (d) pressão na entrada da linha nas corridas; (e) espaço–tempo do
    fluxo no ciclo de relaxação."""
    mp, p = _toy()
    A, W, WJ = _toy_scan()
    runs = {}
    for a in mp.A_RUNS:
        path = fl.RUNS / "paper01" / "toy" / f"a{a:.2f}.npz"
        if path.exists():
            with np.load(path) as d:
                runs[a] = (d["T"], d["X"])
    N = p["N"]
    fig = plt.figure(figsize=(fl.DOUBLE, 0.72 * fl.DOUBLE))
    gs = fig.add_gridspec(2, 6, hspace=0.5, wspace=1.15)
    ax_f, ax_g, ax_t = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])
    ax_p, ax_x = fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])
    shades = dict(zip(mp.A_RUNS, plt.get_cmap("cividis")(np.linspace(0.0, 0.85, len(mp.A_RUNS)))))
    # (a) plano de fase: característica f(y) + a y (tracejada) e ciclo simulado (cheio)
    y = np.linspace(-0.6, 2.6, 300)
    for a in mp.A_RUNS:
        ax_f.plot(y, mp.f(y, p) + a * y, color=shades[a], lw=0.8, ls=(0, (3, 2)))
        if a in runs:
            T, X = runs[a]
            late = T > T[-1] - 40.0
            ax_f.plot(X[-1][late], X[N - 1][late], color=shades[a], lw=1.1, label=f"$a = {a:g}$")
            ax_f.plot(X[-1][-1], X[N - 1][-1], "o", ms=2.5, color=shades[a])
    ax_f.set_ylim(0.8, 3.0)
    ax_f.set_xlabel("terminal flux $y$")
    ax_f.set_ylabel("pressure $p(1)$")
    ax_f.legend(fontsize=6, loc="upper left")
    fl.panel_label(ax_f, "(a)")
    # (b) crescimento dos modos lentos
    cplx = np.abs(W[:, 0].imag) > 1e-9
    real2 = (~cplx) & (np.abs(W[:, 1].imag) < 1e-9)
    ax_g.plot(A, W[:, 0].real, color=fl.INK, lw=1.1, label="with $\\mathbf{M}$: slow modes")
    ax_g.plot(A, np.where(real2, W[:, 1].real, np.nan), color=fl.INK, lw=1.1, ls=(0, (4, 2)))
    ax_g.plot(A, WJ.real, color=fl.COLORS["primitive"], lw=1.0, label="$\\mathbf{J}$ alone: leading")
    ax_g.axhline(0, color=fl.INK, lw=0.6)
    a_h = _crossing(A[::-1], W[::-1, 0].real)
    low = A < a_h if np.isfinite(a_h) else np.zeros(A.size, bool)
    a_r = A[np.flatnonzero(low & ~cplx)[-1]] if (low & ~cplx).any() else np.nan  # colisão no eixo real
    for v, lab in ((a_h, "$a_H$"), (a_r, "$a_R$")):
        if np.isfinite(v):
            for ax in (ax_g, ax_t):
                ax.axvline(v, color=fl.GRID, lw=0.8)
            ax_g.text(v, 1.0, lab, transform=ax_g.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5)
    ax_g.set_xlim(0.2, 1.3)
    ax_g.set_xlabel("terminal resistance $a$")
    ax_g.set_ylabel("Re $\\lambda$")
    ax_g.legend(fontsize=5.8, loc="upper right")
    fl.panel_label(ax_g, "(b)")
    # (c) período do par e dos ciclos simulados
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(cplx, 2 * np.pi / np.abs(W[:, 0].imag), np.nan)
    ax_t.plot(A, per, color=fl.INK, lw=1.1, label="slow pair")
    for a, (T, X) in runs.items():
        late = T > T[-1] - 150.0
        yy = X[-1][late] - X[-1][late].mean()
        up = np.flatnonzero((yy[:-1] < 0) & (yy[1:] >= 0))
        if up.size >= 2 and np.ptp(X[-1][late]) > 1e-3:
            ax_t.plot(a, (T[late][up[-1]] - T[late][up[0]]) / (up.size - 1), "o", ms=4, color=shades[a])
    ax_t.plot([], [], "o", ms=4, color="0.5", label="simulated cycles")
    ax_t.set_ylim(0, 20)
    ax_t.set_xlabel("terminal resistance $a$")
    ax_t.set_ylabel("period")
    ax_t.set_xlim(0.2, 1.3)
    ax_t.legend(fontsize=6, loc="lower left")
    fl.panel_label(ax_t, "(c)")
    # (d) transitórios
    for a, (T, X) in runs.items():
        late = T > T[-1] - 30.0
        ax_p.plot(T[late] - T[late][0], X[0][late], color=shades[a], lw=1.0, label=f"$a = {a:g}$")
    ax_p.set_xlabel("time")
    ax_p.set_ylabel("pressure $p(0)$")
    ax_p.legend(fontsize=6, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False)
    fl.panel_label(ax_p, "(d)")
    # (e) espaço–tempo do fluxo na relaxação
    a_rel = min(runs) if runs else None
    if a_rel is not None:
        T, X = runs[a_rel]
        late = T > T[-1] - 30.0
        Wf = np.vstack([p["k"] * (p["Pr"] - X[0][late]), X[N : 2 * N - 1][:, late], X[-1][late][None, :]])
        im = ax_x.pcolormesh(
            T[late] - T[late][0], np.arange(N + 1) / N, Wf, cmap="Blues", shading="nearest", rasterized=True
        )
        fig.colorbar(im, ax=ax_x, label="flux $w$", fraction=0.05)
        ax_x.set_title(f"$a = {a_rel:g}$", fontsize=7.5)
    ax_x.set_xlabel("time")
    ax_x.set_ylabel("position $x$")
    fl.panel_label(ax_x, "(e)")
    fl.save(fig, out, "model_problem", png)


# ------------------------------------------------------------------------------------------------ apêndice de reprodução
def _tex(v):
    """Texto de uma chave ou valor JSON para LaTeX (\texttt com os caracteres especiais escapados)."""
    t = str(v).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("{", "\\{").replace("}", "\\}")
    return "\\texttt{" + t.replace('"', "") + "}"


def _base_case_dict():
    from mfsim.io import load_json

    return load_json(str(BASE_CASE))


def tab_geometry(out):
    """Troços do caso base (poço e linha com riser): posição ao longo do duto, comprimento, inclinação e células, a
    partir dos pontos do JSON, mais as variantes de malha do estudo de malha."""
    d = _base_case_dict()
    pipes = {e["id"]: e for e in d["network"]["edges"] if "geometry" in e}
    names = {
        "well": ["well"],
        "1": ["flowline, horizontal", "flowline, descent"] + [f"bend, chord {k}" for k in range(1, 5)] + ["riser", "top section"],
    }
    rows = []
    for pid, label in (("well", "well"), ("1", "flowline--riser")):
        e = pipes[pid]
        pts = np.array([q["xyz"] for q in e["geometry"]["points"]], float)
        segs = e["geometry"]["segments"]
        s0 = 0.0
        for k, sg in enumerate(segs):
            a, b = pts[k], pts[k + 1]
            L = float(np.hypot(b[1] - a[1], b[2] - a[2]))
            ang = float(np.degrees(np.arctan2(b[2] - a[2], b[1] - a[1])))
            nm = names[pid][k] if k < len(names[pid]) else f"segment {k + 1}"
            rows.append([f"{label}: {nm}", f"{e['D'] * 1e3:g}", f"{s0:.1f}", f"{L:.1f}", f"{b[1]:.1f}", f"{b[2]:.1f}", f"{ang:+.1f}", str(sg["n"])])
            s0 += L
    fl.tex_table(
        out,
        "tab_geometry",
        ["pipe: section", "$D$ (mm)", "start $s$ (m)", "length (m)", "end $x$ (m)", "end $z$ (m)", "inclination (deg)", "cells"],
        rows,
        align="lrrrrrrr",
        notes=(
            "Coordinates with $x$ horizontal and $z$ the elevation above sea level; the well goes from its bottom to the "
            "wellhead at the start of the flowline. The bend is a circular arc of \\SI{5}{\\metre} radius represented by "
            "four chords. Wall roughness: \\SI{0.2}{\\milli\\metre} in the well, \\SI{0.6}{\\milli\\metre} in the "
            "flowline and riser. Mesh variants (Section~\\ref{sec:mesh}): coarse, 8 cells in the well and 13, 5, 4, 8, 2 "
            "in the sections of the flowline--riser pipe; fine, 30 in the well and 50, 20, 8, 32, 2."
        ),
    )


def tab_parameters(out):
    """Todos os parâmetros do caso base com unidades, lidos do JSON e do modelo construído (constantes do código
    incluídas: larguras de chaveamento, banda do limitador, limites da projeção)."""
    from mfsim.simulation import build_model, state_bounds

    d = _base_case_dict()
    net = _base_network()
    model = build_model(net)
    sch, pr = model.pipes[0].scheme, model.pipes[0].model.params
    fd, res = d["fluid"], next(e for e in d["network"]["edges"] if e.get("type") == "reservoir")
    mk, ms, pvt = d["meta"]["market"], d["meta"]["mass"], d["meta"]["pvt"]
    ch = next(e for e in d["network"]["edges"] if e.get("type") == "choke")
    integ, ref = d["integration"], d["reference"]
    sr = ref["state_ref"]
    lo, hi = state_bounds(model)
    a_lo = float(lo[model.pipes[0].offset + np.flatnonzero(model.pipes[0].free % 4 == 0)[0]])
    K = 0.0075 * ch["CV"]
    rows = [
        [
            "fluids",
            "liquid",
            f"$\\rho_L = {fd['rhoL0']:g}\\,(P/P_0)^{{1/\\gamma_L}}$ kg/m$^3$, $\\gamma_L = {fd['gammaL']:g}$, $\\mu_L = \\num{{{fd['muL']:g}}}$ Pa\\,s",
        ],
        [
            "",
            "gas",
            f"$\\rho_G = {fd['rhoG0']:g}\\,(P/P_0)^{{1/\\gamma_G}}$ kg/m$^3$, $\\gamma_G = {fd['gammaG']:g}$, $\\mu_G = \\num{{{fd['muG']:g}}}$ Pa\\,s; $P_0 = {fd['P0'] / 1e5:g}$ bar",
        ],
        ["", "surface tension", f"$\\sigma = {fd['sigma']:g}$ N/m"],
        [
            "",
            "PVT (standard conditions {pvt['std']})",
            f"$R_s = {pvt['Rs']:g}$ and $\\mathrm{{GOR}}_{{fluid}} = {pvt['GOR']:g}$ m$^3$/m$^3$ at {pvt['P_ref_bar']:g} bar and {pvt['T_ref_C']:g} \\si{{\\celsius}}; $\\rho_o = {pvt['rho_o_std']:g}$, $\\rho_{{gd}} = {pvt['rho_gd_std']:g}$, $\\rho_{{gf}} = {pvt['rho_gf_std']:g}$ kg/m$^3$",
        ],
        [
            "well",
            "inflow",
            f"$J = {mk['PI_Sm3_d_bar']:g}$ m$^3$/(d\\,bar), $\\mathrm{{GOR}} = {mk['GOR_Sm3_Sm3']:g}$ m$^3$/m$^3$, $P_{{res}} = {mk['P_res_bar']:g}$ bar; $w_G = {ms['w_gas']:.4g}$, $J_L = {res['IPL']:.4g}$, $J_G = \\num{{{res['IPG']:.4g}}}$ kg/(s\\,bar); $q_k = J_k \\max(0, P_{{res}} - P_{{wf}})$",
        ],
        [
            "",
            "choke",
            f"$\\Delta P = \\dot m Q/(K a)^2$ [bar], $K = 0.0075\\,c_v$, $c_v = {ch['CV']:g}$ ($K a$ = {1e4 * K / np.sqrt(1e5):.2f} cm$^2$ at $a = 1$); $|Q|$ regularised with $10^{{-6}}$ m$^3$/s",
        ],
        ["", "separator", f"{next(v for v in d['network']['vertices'] if v['id'] == '2')['spec']['PT'] * ref['PR'] / 1e5:g} bar"],
        ["", "wellhead", "connector: the pressure of the flowline inlet is imposed on the well outlet, the well flow rates on the flowline inlet"],
        [
            "",
            "pipe ends",
            "well inlet: flow rates from the inflow law, zero gradient of $\\alpha_G$; riser outlet: pressure from the choke law",
        ],
        [
            "",
            "intermittent",
            "slug unit with the film at equilibrium, Zabaras frequency and Stuhmiller interfacial pressure with the cell slip (Appendix~\\ref{app:closures})",
        ],
        [
            "closures",
            "evaluation bounds",
            f"$\\alpha_G$ cut to [{pr.alpha_min:g}, {pr.alpha_max:g}] inside the closures",
        ],
        [
            "",
            "stratified",
            (
                f"interfacial factor of Andritsos and Hanratty with $v_{{GT}} = {pr.v_gt:g}$ m/s"
                if hasattr(pr, "v_gt")
                else "interfacial factor of Andritsos and Hanratty with $v_{GT} = 5.5$ m/s"
            ),
        ],
        [
            "",
            "bubbly",
            "physical drag (terminal velocity with $g$ and $\\sigma/\\rho_L$); $\\Delta P_L = 0.25\\rho_L v_r^2$, $|v_r| \\leq 3$ m/s",
        ],
        [
            "",
            "virtual mass",
            f"$C_{{vm}} = {pr.virtual_mass:g}$, $\\lambda = {getattr(pr, 'vm_lambda', 1.0):g}$, weighted by the bubbly weight $w_B$",
        ],
        ["", "interfacial pressure", "Bestion term off ($\\delta = 0$)"],
        [
            "flow patterns",
            "identification",
            "automatic, smooth and local; logistic switches with $k = 20$ (Appendix~\\ref{app:patterns})",
        ],
        [
            "",
            "low-void rule",
            f"bubbly closures below $\\alpha_G = {pr.low_void_alpha:g}$, $C^1$ ramp up to {2 * pr.low_void_alpha:g}",
        ],
        [
            "",
            "local rules",
            "annular above $\\alpha_G = 0.9$; no stratified within \\SI{10}{\\degree} of vertical; film bridging with the cell liquid fraction",
        ],
        [
            "scheme",
            "staggered FV",
            f"conservative variables $(\\rho_G\\alpha_G, \\rho_L\\alpha_L)$; face closures with the donor fraction",
        ],
        [
            "",
            "donor switch",
            f"$w = \\tfrac12[1 + \\tanh(q/\\epsilon)]$, $\\epsilon = {sch.switch_frac:g}\\max|q|_{{\\mathrm{{nb}}}} + q_s$, $q_{{s,L}} = \\num{{{sch.q_switch_L:.3g}}}$, $q_{{s,G}} = \\num{{{sch.q_switch_G:.3g}}}$ kg/s",
        ],
        [
            "",
            "flux limiter",
            f"$C^1$ ramp of the donor phase fraction over [0, {sch.flux_band:g}], donor switch with $\\epsilon = {sch.flux_eps:g}\\,q_s$",
        ],
        ["", "vanishing phase", f"relaxed to the no-slip flow rate with $\\tau = {sch.relax_time:g}$ s"],
        ["", "artificial viscosity", f"$\\nu = C c h$, $C = {sch.art_visc:g}$"],
        ["", "initial state", "blocked start of the case generator (Section~\\ref{sec:case}), stored at the nodes of the scheme"],
        ["integrator", "formula", f"BDF of order 1--{integ['max_order']}, variable step, modified Newton"],
        [
            "",
            "tolerances",
            f"rtol = {integ['rtol']:g}; atol = {integ['tol_range']:g} $\\times$ (1, {sr['PT'] * ref['PR'] / 1e5:.2f} bar, {sr['qL']:.3g} kg/s, {sr['qG']:.3g} kg/s) for ($\\alpha_G$, $P$, $q_L$, $q_G$)",
        ],
        ["", "Newton", f"tolerance {integ['newton_tol']:g} (scaled), at most {integ['max_newton']} iterations"],
        [
            "",
            "steps",
            f"first \\num{{{integ['first_step']:g}}} s, minimum \\num{{{integ['min_step']:g}}} s, maximum none (\\SI{{1}}{{\\second}} from an equilibrium)",
        ],
        [
            "",
            "projection",
            f"$\\alpha_G \\in [{a_lo:g}, {float(hi[model.pipes[0].offset + np.flatnonzero(model.pipes[0].free % 4 == 0)[0]]):g}]$, order 1 after a projection",
        ],
        [
            "",
            "backend",
            f"{integ['backend'].upper() if integ['backend'] == 'jax' else integ['backend']}, exact Jacobian; boundary fluxes integrated with the same recurrence",
        ],
    ]
    fl.tex_table(
        out,
        "tab_parameters",
        ["group", "item", "value"],
        rows,
        align="llp{0.66\\linewidth}",
        notes="Values read from the case file of the application and from the simulator (Section~\\ref{sec:case}).",
    )


def tab_runs(out):
    """Corridas de cada estudo: lote, partida, tempo simulado, saída, passo máximo, orçamento e número de corridas,
    lidos dos ficheiros de lote."""

    def info(path):
        b = json.load(open(fl.PROJECT / path))
        sets = b["runs"][0].get("set", [])  # a primeira entrada (a de referência de cada lote)
        get = lambda key: next((x.split("=", 1)[1] for x in sets if x.startswith(key + "=")), None)  # noqa: E731
        return len(b["runs"]), get

    rows = []
    specs = [
        ("choke openings (\\ref{sec:opening})", "batches/paper01_offshore_dynamics.json", "blocked", "8000", "2"),
        ("gas--oil ratio (\\ref{sec:gor})", "batches/paper01_offshore_gor.json", "blocked", "8000", "2"),
        ("ablation (\\ref{sec:ablation})", "batches/paper01_offshore_ablation.json", "blocked", "4000", "2"),
        ("mesh (\\ref{sec:mesh})", "batches/paper01_offshore_mesh.json", "blocked", "4000", "2"),
        ("backends (\\ref{sec:backend})", "batches/paper01_offshore_backend.json", "blocked", "2000", "2"),
    ]
    for lab, path, start, tf, dt in specs:
        try:
            n, get = info(path)
        except FileNotFoundError:
            rows.append([lab, start, tf, dt, "--", "--", "--"])
            continue
        ms = get("integration.max_step")
        bud = get("integration.max_cpu")
        rows.append([lab, start, tf, dt, ms or "none", f"{float(bud):.0f}" if bud else "none", str(n)])
    rows.append(
        [
            "model problem (\\ref{sec:toy})",
            "steady state + 0.1 \\%",
            "300",
            "0.1",
            "0.5",
            "none",
            "4",
        ]
    )
    fl.tex_table(
        out,
        "tab_runs",
        ["study (section)", "start", "time (s)", "output (s)", "max.\\ step (s)", "budget (s)", "runs"],
        rows,
        align="lllrrrr",
        notes=(
            "Each run is the application case of Table~\\ref{tab:parameters} with the changes of its study (choke "
            "opening, gas--oil ratio, mesh, or the variants of Table~\\ref{tab:ablationchanges}). Budget: wall-clock "
            "time after which a run is stopped with its partial trajectory. The model problem uses rtol $= 10^{-4}$ "
            "and atol $= 10^{-6}$ (dimensionless units)."
        ),
    )


ABL_CHANGES = {
    "reference": "the configuration of Table~\\ref{tab:parameters}",
    "primitive": "the integrator advances the primitive variables $(\\alpha_G, P)$ instead of the masses per unit volume; vanishing phase relaxed to a zero flow rate (the no-slip target needs the flux limiter of the conservative form)",
    "relax_raw": "vanishing phase relaxed to a zero flow rate instead of the no-slip flow rate",
    "no_flux_limiter": "no donor-cell limiter of the mass fluxes",
    "no_art_visc": "$C = 0$ in the artificial viscosity",
    "closure_mean": "face closures evaluated with the arithmetic mean of the void fractions of the two neighbouring cells",
    "ipe_thesis": "pattern criteria evaluated on the equilibrium level and void fraction instead of the cell state, with sharp switches ($k = 10^6$)",
    "ipe_smooth_only": "pattern criteria evaluated on the equilibrium level and void fraction instead of the cell state ($k = 20$)",
    "no_low_void": "low-void threshold $10^{-12}$ (rule inactive)",
    "no_virtual_mass": "$C_{vm} = 0$",
    "bubble_drag_thesis": "bubble terminal velocity $v_b = (0.51\\,d_b + 2.14/d_b)^{1/2}$, without $g$ and $\\sigma/\\rho_L$",
    "unit_consistent": "slug unit of the thesis with the slug shortened so that the unit conserves the mass of the cell (film cut at $\\alpha_f = 0.95$)",
    "unit_thesis": "slug unit with the slug length of Norris and the film holdup from the mass balance, cut at $\\alpha_f = 0.95$",
    "freq_thesis": "slug frequency $\\omega_s = (u_{GS}/D)[0.0323/2\\,(u_{LS}/u_{GS})(\\rho_L/\\Delta\\rho)^{1/2} f_L^{-1/2} Eo^{-0.2}]^{n}$, $n = N_f/(260 + 0.85 N_f)$",
    "k_switch_80": "logistic switches with $k = 80$ instead of 20",
    "bdf1": "BDF of order 1 only",
    "uniform_atol": "absolute tolerance $10^{-6}$ for every state, with the pressure in MPa and the flow rates in kg/s",
    "newton12": "at most 12 Newton iterations per step",
    "scipy_bdf": "variable-order BDF of SciPy on the primitive form, vanishing phase relaxed to a zero flow rate, rtol $= 3\\times10^{-3}$, atol $= 10^{-3}$ for every state, maximum step \\SI{1}{\\second}, no projection on bounds",
    "gammaL1000": "$\\gamma_L = 1000$",
    "valve_thesis": "choke density weighted by mass, $\\Delta P = \\dot m|\\dot m|/(\\rho_M (K a)^2)$ with $\\rho_M = (q_L\\rho_L + q_G\\rho_G)/(q_L + q_G)$",
    "sharp_bend": "no bend: the descending line ends at $x = 319.59$ m, $z = -15.70$ m, where a vertical riser of \\SI{20}{\\metre} (5 cells) starts",
}


def tab_ablation_changes(out):
    """Variantes da ablação descritas em termos do modelo (o código não é público: nada de chaves de ficheiro)."""
    b = json.load(open(fl.PROJECT / "batches" / "paper01_offshore_ablation.json"))
    rows = [[ABL_LABELS.get(r["name"], r["name"]), ABL_CHANGES.get(r["name"], "--")] for r in b["runs"]]
    fl.tex_table(
        out,
        "tab_ablationchanges",
        ["variant", "change to the configuration of the application"],
        rows,
        align="lp{0.6\\linewidth}",
        notes="Every variant has the choke wide open ($a = 1$) at the base point and runs for \\SI{4000}{\\second}.",
    )


def tab_analysis(out):
    """Definições das grandezas derivadas das corridas, com as constantes lidas do código."""
    import inspect

    from mfsim.analysis import N_EFF_GLOBAL, SLOW_MODE, cycle_class, jansen_choke_dp_crit

    cc = inspect.signature(cycle_class).parameters
    jj = inspect.signature(jansen_choke_dp_crit).parameters
    rows = [
        [
            "cycle class",
            f"stable if the range of the smoothed $P_{{in}}$ is below max({cc['amp_stable'].default:g} bar, {100 * cc['amp_rel'].default:g} \\% of its mean) or it crosses its mean at most once; severe slugging if the liquid outflow peaks at {cc['surge'].default:g} times its mean or more; pressure cycle otherwise",
        ],
        [
            "smoothing, period",
            f"centred moving average of {SMOOTH:g} s on $P_{{in}}$; period = mean time between up-crossings of the mean",
        ],
        [
            "windows",
            "dense runs: last \\SI{3000}{\\second}; map: windows of \\SI{2000}{\\second}, stop when the range is below \\SI{0.1}{\\bar} or two windows agree within \\SI{10}{\\percent} in range and period, at most \\SI{8000}{\\second}",
        ],
        [
            "blowdowns (tables)",
            "minima of $P_{in}$ below the middle of its 2--98 \\% percentile range, after \\SI{300}{\\second}, at least \\SI{150}{\\second} apart",
        ],
        [
            "fronts",
            "seal: liquid-filled length upstream of the low point ($\\alpha_G < 0.5$); level: top of the liquid column in the riser ($\\alpha_G < 0.5$); riser full: level above 98 \\% of the riser for \\SI{20}{\\second}; blowout start: maximum of the smoothed $P_{in}$",
        ],
        [
            "linear stability",
            f"eigenvalues of $\\bJ\\bv = \\lambda\\bM\\bv$ with the exact Jacobian; global slow modes: $|\\lambda| < {SLOW_MODE:g}$ s$^{{-1}}$ and $n_{{\\mathrm{{eff}}}} \\geq {N_EFF_GLOBAL:g}$, $n_{{\\mathrm{{eff}}}} = 1/\\sum_j p_j^2$ with $p_j$ the share of node $j$ in $\\|\\bv\\|^2$, states scaled by their tolerance ranges; class from the leading global slow mode",
        ],
        [
            "continuation",
            "Newton on $\\bF = 0$ (residual $10^{-10}$, at most 60 iterations) from the neighbouring point; at \\SI{15}{\\bar} steps of 0.01 in $a$ up to 0.4 and 0.05 above; map: neighbouring grid points, repeated passes; $a_H$ by linear interpolation of Re $\\lambda$ in $\\log a$",
        ],
        [
            "choke criterion",
            f"$\\Delta p_c$ of \\citet{{jansen1996}} with $\\alpha = {jj['alpha'].default:g}$, $\\alpha_p = {jj['alpha_p'].default:g}$, $L = {jj['L'].default:g}$ m, $H = {jj['H'].default:g}$ m, $P_S = {jj['P_S'].default / 1e5:g}$ bar, mean of $\\Phi = 1$ and 0.87; choke $\\Delta p = q_L^2/(\\rho_L (K a)^2)$ with the production of the steady state",
        ],
        [
            "well-posedness",
            "eigenvalues of $\\bA$ in every cell, one state every \\SI{5}{\\second}; ill-posed if the largest $|\\mathrm{Im}\\,\\lambda| > 10^{-6}$ m/s",
        ],
        ["mass error", "inventory change minus the boundary fluxes integrated with the recurrence of the integrator"],
    ]
    fl.tex_table(out, "tab_analysis", ["quantity", "definition"], rows, align="lp{0.74\\linewidth}")


def software_note(out):
    """Versões do software e revisão do repositório com que o PDF foi gerado (fragmento LaTeX)."""
    import platform
    import subprocess

    import jax
    import scipy

    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=fl.PROJECT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        rev = "--"
    txt = (
        f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__} and "
        f"JAX {jax.__version__}."
    )
    path = Path(out) / "software.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("% gerado pelo guião de figuras do artigo; não editar à mão\n" + txt + "\n")
    print(f"tabela: {path}")


FIGURES = {
    "geometry": geometry,
    "hyperbolicity": hyperbolicity,
    "steady_profiles": steady_profiles,
    "spacetime": spacetime,
    "cycle_anatomy": cycle_anatomy,
    "cycle_profiles": cycle_profiles,
    "stability_map": stability_map,
    "bifurcation": bifurcation,
    "discretisation": discretisation,
    "linear_stability": linear_stability,
    "model_problem": model_problem,
    "gor": gor,
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
    "tab_gor": tab_gor,
    "tab_hopf_mesh": tab_hopf_mesh,
    "tab_geometry": tab_geometry,
    "tab_parameters": tab_parameters,
    "tab_runs": tab_runs,
    "tab_ablationchanges": tab_ablation_changes,
    "tab_analysis": tab_analysis,
    "software": software_note,
}

if __name__ == "__main__":
    fl.run_cli(FIGURES, __doc__.splitlines()[0], tables=TABLES)
