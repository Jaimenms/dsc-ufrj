"""Receitas de condição inicial dos casos de intermitência severa (o código que vivia em
`severe_slug_case`), como funções da malha e de parâmetros numéricos resolvidos.

- `blocked_ic`: fase de bloqueio: bolsão de gás estratificado na linha até `s_front`,
  líquido daí até ao topo do riser, pressão hidrostática a partir do separador com salto
  `dP_margin` na frente de líquido; vazão nominal de líquido em todo o duto e de gás só no
  bolsão.
- `shut_in_ic`: fluido parado, fração de gás `alpha_line` antes de `s_elbow0` e `alpha_riser`
  depois, pressão hidrostática da mistura a partir do separador.

A forma JSON (`"ic": {"recipe": ..., parâmetros}`) guarda os valores já resolvidos (por
exemplo `pocket_alpha` calculado do equilíbrio), para o ficheiro ser auto-contido.
"""

from __future__ import annotations

import numpy as np

from ..core.slots import G


def blocked_ic(s_nodes, z_nodes, *, s_front, s_top, pocket_alpha, P_out, dP_margin, qL, qG, fluid, PR):
    """Arrays (n_nodes, 4) [alphaG, PT, qL, qG] da partida bloqueada; devolve também P e alphaG."""
    s_nodes = np.asarray(s_nodes, float)
    z_nodes = np.asarray(z_nodes, float)
    n_nodes = s_nodes.size
    aG0 = np.where(s_nodes < s_front - 1e-9, pocket_alpha, 0.0)
    aG0 = np.where(s_nodes > s_top + 1e-9, 0.0, aG0)  # horizontal de saída
    P = np.empty(n_nodes)
    P[-1] = P_out
    for i in range(n_nodes - 2, -1, -1):
        rhoL_, rhoG_, _, _ = fluid.props(P[i + 1])
        if s_nodes[i] < s_front - 1e-9:
            rho_h = rhoG_  # no bolsão estratificado a pressão (do gás) é quase uniforme
        else:
            a = 0.5 * (aG0[i] + aG0[i + 1])
            rho_h = (1 - a) * rhoL_ + a * rhoG_
        P[i] = P[i + 1] + rho_h * G * (z_nodes[i + 1] - z_nodes[i])
        if s_nodes[i] < s_front - 1e-9 <= s_nodes[i + 1]:
            P[i] += dP_margin  # salto de pressão na frente de líquido
    qG0 = np.where(s_nodes < s_front - 1e-9, qG, 0.0)
    return np.stack([aG0, P / PR, np.full(n_nodes, qL), qG0], axis=1), P, aG0


def shut_in_ic(s_nodes, z_nodes, *, s_elbow0, alpha_line, alpha_riser, P_out, fluid, PR):
    """Arrays (n_nodes, 4) da partida de linha fechada (fluido parado)."""
    s_nodes = np.asarray(s_nodes, float)
    z_nodes = np.asarray(z_nodes, float)
    n_nodes = s_nodes.size
    aG0 = np.where(s_nodes < s_elbow0 - 1e-9, alpha_line, alpha_riser)
    P = np.empty(n_nodes)
    P[-1] = P_out
    for i in range(n_nodes - 2, -1, -1):
        rhoL_, rhoG_, _, _ = fluid.props(P[i + 1])
        a = 0.5 * (aG0[i] + aG0[i + 1])
        rho_mix = (1 - a) * rhoL_ + a * rhoG_
        P[i] = P[i + 1] + rho_mix * G * (z_nodes[i + 1] - z_nodes[i])
    return np.stack([aG0, P / PR, np.zeros(n_nodes), np.zeros(n_nodes)], axis=1)


RECIPES = {"blocked": blocked_ic, "shut_in": shut_in_ic}


def materialize(recipe: dict, points, n_elements, grading, fluid, PR):
    """Arrays de condição inicial de um duto a partir da receita JSON e da sua malha."""
    from ..geometry import pipe_mesh

    pts = np.asarray(points, float)
    mesh = pipe_mesh(
        pts[:, 0], pts[:, 1], pts[:, 2], np.asarray(n_elements, int), grading=None if grading is None else list(grading)
    )
    s, z = mesh["s"], mesh["xyz"][:, 2]
    r = dict(recipe)
    kind = r.pop("recipe")
    if kind == "blocked":
        ic, _, _ = blocked_ic(
            s,
            z,
            s_front=float(r["s_front"]),
            s_top=float(r["s_top"]),
            pocket_alpha=float(r["pocket_alpha"]),
            P_out=float(r["P_out_bar"]) * 1e5,
            dP_margin=float(r["dP_margin_bar"]) * 1e5,
            qL=float(r["qL"]),
            qG=float(r["qG"]),
            fluid=fluid,
            PR=PR,
        )
        return ic
    if kind == "shut_in":
        return shut_in_ic(
            s,
            z,
            s_elbow0=float(r["s_elbow0"]),
            alpha_line=float(r.get("alpha_line", 0.5)),
            alpha_riser=float(r.get("alpha_riser", 0.01)),
            P_out=float(r["P_out_bar"]) * 1e5,
            fluid=fluid,
            PR=PR,
        )
    raise ValueError(f"receita de condição inicial desconhecida: {kind}")
