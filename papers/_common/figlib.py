"""Apoio às figuras dos artigos (importado por `papers/<artigo>/figures/make_figures.py`).

Dá os caminhos do projeto (corridas em `runs/` e `notebooks/data/`), o estilo matplotlib de artigo,
as larguras de coluna da Elsevier, uma paleta categórica em ordem fixa por entidade e a gravação em
PDF (vetorial, lido pelo LaTeX) mais PNG (pré-visualização). Os guiões de figuras só leem corridas
gravadas pelos lotes (`mfsim batch`) e pelas ferramentas de `tools/`; não simulam. Grandezas que se
calculam do modelo sem integrar no tempo (autovalores, mapas de hiperbolicidade) ficam em cache em
`runs/<artigo>/cache/`.

Trazido do research-factory (01-multiphase-compositional-flow/papers/_common) em 2026-10-16 e
adaptado ao mfsim: corridas pelo `mfsim.runs.load_run`, paleta por entidade do escoamento bifásico.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COMMON = Path(__file__).resolve().parent
PROJECT = COMMON.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

RUNS = Path(os.environ.get("MFSIM_RUNS", PROJECT / "runs"))
DATA = PROJECT / "notebooks" / "data"

MM = 1 / 25.4
SINGLE, MID, DOUBLE = 90 * MM, 140 * MM, 190 * MM  # larguras de figura da Elsevier, em polegadas

# Paleta categórica (Okabe–Ito, segura para daltonismo; validada com o validador do skill dataviz,
# pares adjacentes e, nos conjuntos de até 4, todos os pares) atribuída por entidade, nunca por posição:
# a mesma entidade tem a mesma cor em todas as figuras de todos os artigos. Entidades de tipos que
# nunca aparecem na mesma figura podem partilhar o tom.
COLORS = {
    # padrões de escoamento (mapas x–t e perfis): 4 tons, todos os pares validados
    "stratified": "#0072B2",
    "intermittent": "#E69F00",
    "bubble": "#009E73",
    "annular": "#D55E00",
    # classes do mapa de estabilidade (todos os pares validados, contraste >= 3:1)
    "stable": "#0072B2",
    "cycling": "#009E73",
    "slugging": "#D55E00",
    # partidas, formulações e backends (pares)
    "steady": "#0072B2",
    "blocked": "#D55E00",
    "conservative": "#0072B2",
    "primitive": "#D55E00",
    "jax": "#0072B2",
    "numpy": "#D55E00",
    # esquemas espaciais (ordem fixa)
    "fv_upwind": "#0072B2",
    "galerkin": "#D55E00",
    "petrov_galerkin": "#009E73",
    "fd_central": "#E69F00",
    "fd_upwind": "#CC79A7",
    "reference": "#000000",
}
PATTERNS = ("stratified", "intermittent", "bubble", "annular")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d2"


def use_style():
    plt.style.use(str(COMMON / "mfsim-paper.mplstyle"))


def figsize(width=SINGLE, ratio=0.72):
    return (width, width * ratio)


def find(name):
    """Caminho de uma corrida: caminho dado, `runs/<nome>.npz`, `runs/**/<nome>.npz` ou `notebooks/data/<nome>.npz`;
    None se não existe (a figura sai com um aviso)."""
    p = Path(name)
    if p.suffix == ".npz" and p.exists():
        return p
    stem = p.name[:-4] if p.name.endswith(".npz") else p.name
    base = str(name)[:-4] if str(name).endswith(".npz") else str(name)
    if "/" in base:  # caminho relativo a runs/: só esse (sem procurar o mesmo nome noutras pastas)
        cand = RUNS / f"{base}.npz"
        return cand if cand.exists() else None
    for cand in (RUNS / f"{base}.npz", DATA / f"{stem}.npz"):
        if cand.exists():
            return cand
    hits = sorted(RUNS.rglob(f"{stem}.npz"))
    return hits[0] if hits else None


def load_run(name):
    """Corrida gravada (`mfsim.runs.Run`) ou None se não existe."""
    from mfsim.runs import load_run as _load_run

    path = find(name)
    return None if path is None else _load_run(str(path))


def load_runs(pattern):
    """Corridas `RUNS/<pattern>` (glob) como dict nome -> Run, por ordem de nome."""
    from mfsim.runs import load_run as _load_run

    return {p.stem: _load_run(str(p)) for p in sorted(RUNS.glob(pattern))}


def panel_label(ax, text):
    """Letra do painel, (a), (b)..., no canto superior esquerdo, fora da área de dados."""
    ax.text(-0.02, 1.02, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold")


def placeholder(ax, text):
    """Aviso no lugar dos dados (corrida em falta), para o PDF compilar na mesma."""
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=7, wrap=True, color="0.35")
    ax.set_xticks([])
    ax.set_yticks([])


def save(fig, out_dir, name, png=True):
    """Grava `out_dir/name.pdf` (e `.png` a 300 dpi) e fecha a figura; devolve o caminho do PDF."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(pdf)
    if png:
        fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    print("figura:", pdf)
    return pdf


def tex_table(out_dir, name, header, rows, align=None, notes=None):
    """Grava `out_dir/name.tex` com um `tabular` booktabs (células já formatadas) e `out_dir/name_note.tex`
    com a nota (vazio se não houver), para a nota ficar fora de um eventual `\\resizebox` no manuscrito."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    align = align or "l" + "r" * (len(header) - 1)
    lines = [
        "% gerado pelo guião de figuras do artigo; não editar à mão",
        f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}",
        "\\toprule",
        " & ".join(header) + r" \\",
        "\\midrule",
    ]
    lines += [(r[0] if isinstance(r, str) else " & ".join(str(c) for c in r) + r" \\") for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    path = out_dir / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n")
    (out_dir / f"{name}_note.tex").write_text(("\\par\\smallskip\\footnotesize " + notes + "\n") if notes else "")
    print("tabela:", path)
    return path


def fmt(v, spec="{:.2f}"):
    """Formata um número ou devolve «--» para None/NaN (corridas em falta)."""
    if v is None:
        return "--"
    try:
        if np.isnan(v):
            return "--"
    except TypeError:
        pass
    return spec.format(v)


def run_cli(figures, description, tables=None):
    """Linha de comando comum: `--out DIR` (obrigatório), `--tables DIR` (fragmentos LaTeX das tabelas),
    `--only NOME` (repetível), `--no-png`.

    `figures` é um dict nome -> função(out_dir, png) que grava a figura com esse nome; `tables`, um dict
    nome -> função(out_dir) que grava `out_dir/nome.tex` (um `tabular` lido por `\\input` no manuscrito).
    """
    tables = tables or {}
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", required=True, type=Path, help="pasta das figuras geradas")
    parser.add_argument("--tables", type=Path, help="pasta dos fragmentos LaTeX das tabelas")
    parser.add_argument(
        "--only", action="append", choices=sorted(figures) + sorted(tables), help="gerar só este item (repetível)"
    )
    parser.add_argument("--no-png", action="store_true", help="não gravar a pré-visualização PNG")
    args = parser.parse_args()
    use_style()
    names = args.only or (sorted(figures) + sorted(tables))
    for name in names:
        if name in figures:
            figures[name](args.out, not args.no_png)
        elif args.tables is not None:
            tables[name](args.tables)
    return args
