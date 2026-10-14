"""Caso de intermitência severa (severe slugging).

Geometria clássica (Taitel 1986; Schmidt et al. 1980): linha horizontal,
trecho descendente (inclinação negativa) que termina na base de um riser
vertical, seguido de um trecho horizontal até o separador (pressão fixa).

Gerador de casos, fora do simulador: `severe_slug_case(**parâmetros)` constrói o grafo
(`mfsim.Network`) a partir de ~30 parâmetros físicos (geometria com arco no joelho e malha
graduada, vazões a partir de vSL/vSG, fluido, partida, opções do modelo e do esquema) e
`generate(**parâmetros)` acrescenta a receita de condição inicial e regista os parâmetros em
`meta.generator_params`; `python -m generators.severe_slug --out x.json --param D=0.05` grava
o JSON. Os ficheiros em `cases/severe_slug/` foram produzidos assim. O simulador só lê JSON.
O critério de Bøe e as velocidades superficiais estão em `mfsim.analysis`.
"""

from __future__ import annotations

import numpy as np

from mfsim.fluid import Fluid
from mfsim.core.slots import BC_ABSENT, BC_DIRICHLET, BC_NEUMANN
from mfsim.network.graph import BoundaryVertex, Network, Pipe, TankVertex, specs

G = 9.80665


def severe_slug_case(
    D=0.1,
    L_h1=500.0,
    L_down=1000.0,
    slope_down_deg=-2.0,
    H_riser=100.0,
    L_h2=200.0,
    n_h1=10,
    n_down=30,
    n_riser=20,
    n_h2=5,
    qL=1.0,
    qG=0.05,
    P_out_bar=20.0,
    fluid=None,
    regime=6,
    method="fv_upwind",
    tf=6000.0,
    dt_out=5.0,
    roughness=50e-6,
    single_pipe=True,
    extra=None,
    start="guess",
    t_ramp=100.0,
    alpha_line0=0.5,
    alpha_riser0=0.01,
    pocket_alpha=None,
    front_liquid_length=0.0,
    dP_margin_bar=0.3,
    elbow_radius=0.0,
    n_elbow=0,
    grading_down=1.0,
    grading_riser=1.0,
    model=1,
    upstream_volume=0.0,
):
    """start:
    "guess"    - estimativa de escoamento estabelecido (initial_guess);
    "shut_in"  - fluido parado, vazões em rampa (t_ramp);
    "blocked"  - estado da fase de bloqueio da intermitência severa: bolsão de
                 gás estratificado na linha (fração pocket_alpha, ou a de
                 equilíbrio), trecho final do descendente (front_liquid_length m)
                 e riser cheios de líquido, pressão hidrostática a partir do
                 separador e bolsão a dP_margin_bar acima da pressão na frente
                 de líquido; vazões nominais de líquido em todo o duto e de gás
                 só no bolsão. Com isso o primeiro blowdown ocorre em ~100 s.
    """
    """Constrói o grafo (`Network`): um único duto com 4 trechos (ou 4 dutos acoplados por
    vértices dinâmicos se single_pipe=False); `net.meta` guarda a geometria e as vazões."""
    fluid = fluid or Fluid(
        rhoL0=850.0, rhoG0=18.0, muL=5e-3, muG=1.5e-5, gammaL=1000.0, gammaG=1.1, P0=20e5, sigma=0.03
    )
    PR = 1e6
    th = np.deg2rad(slope_down_deg)
    dy_down = L_down * np.cos(th)
    dz_down = L_down * np.sin(th)
    # traçado (y horizontal, z vertical)
    yC, zC = L_h1 + dy_down, dz_down  # joelho (base do riser)
    if elbow_radius > 0 and n_elbow > 0:
        # joelho em arco de raio R tangente ao trecho descendente e ao riser: os
        # elementos do arco têm ângulos intermediários entre slope_down e 90°, o que
        # dá sentido à transição suave dos fechamentos por ângulo (vertical_blend)
        R = elbow_radius
        t_tan = R * (1.0 - np.sin(th)) / np.cos(th)  # distância do joelho aos pontos de tangência
        y1, z1 = yC - t_tan * np.cos(th), zC - t_tan * np.sin(th)
        ang = th + (np.pi / 2 - th) * np.arange(1, n_elbow + 1) / n_elbow
        y_arc = y1 + R * (np.sin(ang) - np.sin(th))
        z_arc = z1 - R * (np.cos(ang) - np.cos(th))
        y = [0.0, L_h1, y1] + y_arc.tolist() + [yC, yC + L_h2]
        z = [0.0, 0.0, z1] + z_arc.tolist() + [zC + H_riser, zC + H_riser]
        n_el_list = [n_h1, n_down] + [1] * n_elbow + [n_riser, n_h2]
        grading = [1.0, grading_down] + [1.0] * n_elbow + [grading_riser, 1.0]
        s_elbow0 = L_h1 + L_down - t_tan  # início do arco (frente de líquido na CI bloqueada)
        # grupos de segmentos por duto no modo rede: linha (horizontal + descendente), pescoço (arco), riser, saída
        groups = [[0, 1], list(range(2, 2 + n_elbow)), [2 + n_elbow], [3 + n_elbow]]
    else:
        y = [0.0, L_h1, yC, yC, yC + L_h2]
        z = [0.0, 0.0, zC, zC + H_riser, zC + H_riser]
        n_el_list = [n_h1, n_down, n_riser, n_h2]
        grading = [
            1.0,
            grading_down,
            grading_riser,
            1.0,
        ]  # malha não uniforme: < 1 refina para o joelho (descendente), > 1 refina a base (riser)
        s_elbow0 = L_h1 + L_down
        groups = [[0, 1], [2], [3]]
    x = [0.0] * len(y)
    if start == "shut_in":
        # partida de linha fechada: vazões sobem em rampa de 0 até o valor nominal
        spec_in = {
            "qL": (np.array([0.0, t_ramp]), np.array([0.0, qL])),
            "qG": (np.array([0.0, t_ramp]), np.array([0.0, qG])),
        }
    else:
        spec_in = {"qL": qL, "qG": qG}
    if upstream_volume > 0:
        # volume de gás a montante (gasoduto comprimível): vértice dinâmico com as vazões de
        # alimentação como entradas externas; o duto recebe alphaG e P do vértice e as
        # vazões na face de entrada passam a ser estados (só no duto único)
        if not single_pipe:
            raise ValueError("upstream_volume só está implementado com single_pipe=True")
        v_in = TankVertex("1", volume=upstream_volume, spec=specs(**spec_in))
    else:
        v_in = BoundaryVertex("1", spec=specs(**spec_in))
    v_out = BoundaryVertex("2", spec=specs(PT=P_out_bar * 1e5 / PR))
    extra = dict(extra or {})
    if model == 2:
        # modelo de deriva no escalonado: o hold-up da condição inicial é o dado (bolsão,
        # riser cheio), não o da correlação com as vazões nominais
        extra.setdefault("ic_holdup", "given")
    if single_pipe:
        bc_in_single = (
            (BC_DIRICHLET, BC_DIRICHLET, BC_ABSENT, BC_ABSENT)
            if upstream_volume > 0
            else (BC_NEUMANN, BC_ABSENT, BC_DIRICHLET, BC_DIRICHLET)
        )
        vertices = [v_in, v_out]
        pipes = [
            Pipe.legacy(
                "1",
                "1",
                "2",
                x,
                y,
                z,
                n_el_list,
                D=D,
                roughness=roughness,
                model=model,
                regime=regime,
                method=method,
                bc_in=bc_in_single,
                bc_out=(BC_ABSENT, BC_DIRICHLET, BC_ABSENT, BC_ABSENT),
                extra=extra,
                grading=grading,
            )
        ]
    else:
        # rede: um duto por grupo de trechos (linha, pescoço, riser, saída), acoplados
        # por vértices dinâmicos (balanço de massa num volume, sem momento); as
        # condições de contorno internas são alphaG e P impostos pelo vértice na
        # entrada do duto a jusante e P imposto pelo vértice na saída do duto a montante
        n_pipes = len(groups)
        regimes = list(regime) if isinstance(regime, (list, tuple)) else [regime] * n_pipes
        vol = extra.pop("vertex_volume", 0.05)
        junctions = [3 + j for j in range(n_pipes - 1)]
        vertices = [v_in] + [TankVertex(str(j), volume=vol, spec=specs(qL=0.0, qG=0.0)) for j in junctions] + [v_out]
        vs = [1] + junctions + [2]
        pipes = []
        for i, g in enumerate(groups):
            seg_y = [y[g[0]]] + [y[k + 1] for k in g]
            seg_z = [z[g[0]]] + [z[k + 1] for k in g]
            bc_in = (
                (BC_NEUMANN, BC_ABSENT, BC_DIRICHLET, BC_DIRICHLET)
                if i == 0
                else (BC_DIRICHLET, BC_DIRICHLET, BC_ABSENT, BC_ABSENT)
            )
            # saída para vértice dinâmico: alphaG do vértice no fantasma (célula doadora em refluxo);
            # saída para o separador: alphaG livre
            bc_out = (
                (BC_DIRICHLET, BC_DIRICHLET, BC_ABSENT, BC_ABSENT)
                if i < n_pipes - 1
                else (BC_ABSENT, BC_DIRICHLET, BC_ABSENT, BC_ABSENT)
            )
            pipes.append(
                Pipe.legacy(
                    str(i + 1),
                    str(vs[i]),
                    str(vs[i + 1]),
                    [0.0] * len(seg_y),
                    seg_y,
                    seg_z,
                    [n_el_list[k] for k in g],
                    D=D,
                    roughness=roughness,
                    model=model,
                    regime=regimes[i],
                    method=method,
                    bc_in=bc_in,
                    bc_out=bc_out,
                    extra=dict(extra),
                    grading=[grading[k] for k in g],
                )
            )
    # faixas esperadas dos estados (para tolerâncias absolutas uniformes, tol_range):
    # os picos de blowdown chegam a ~20x a alimentação de líquido e ~50x a de gás
    state_ref = dict(qL=20.0 * qL, qG=50.0 * qG, PT=(P_out_bar * 1e5 + 1.1 * fluid.rhoL0 * G * H_riser + 2e5) / PR)
    net = Network.build(
        "severe_slug",
        fluid,
        vertices,
        pipes,
        t0=0.0,
        tf=tf,
        dt_out=dt_out,
        PR=PR,
        integrator="implicit",
        quadrature_order=2,
        max_step=np.inf,
        atol=1e-6,
        rtol=1e-4,
        state_ref=state_ref,
    )
    if start == "blocked":
        from mfsim.geometry import pipe_mesh

        mesh = pipe_mesh(x, y, z, n_el_list, grading=grading)
        s_nodes = mesh["s"]
        z_nodes = mesh["xyz"][:, 2]
        n_nodes = s_nodes.size
        s_riser0 = s_elbow0  # ponto baixo: início do arco (ou o joelho)
        s_top = mesh["L"] - L_h2  # topo do riser
        s_front = s_riser0 - front_liquid_length
        if pocket_alpha is None:
            # fração de gás de equilíbrio do estratificado no trecho descendente
            from mfsim.closures import ClosureInputs, equilibrium_fractions
            from mfsim.geometry import stratified_level

            P_est = P_out_bar * 1e5 + fluid.rhoL0 * G * H_riser
            rhoL_, rhoG_, CL_, CG_ = fluid.props(P_est)
            Area = np.pi * D**2 / 4
            vLS = qL / rhoL_ / Area
            vGS = qG / rhoG_ / Area
            hD, beta = stratified_level(np.array([0.5]))
            inp = ClosureInputs(
                alphaG=np.array([0.5]),
                alphaL=np.array([0.5]),
                vL=np.array([2 * vLS]),
                vLS=np.array([vLS]),
                vG=np.array([2 * vGS]),
                vGS=np.array([vGS]),
                AreaL=np.array([Area / 2]),
                AreaG=np.array([Area / 2]),
                D=np.array([D]),
                eps=np.array([roughness]),
                theta=np.array([th]),
                rhoL=np.array([rhoL_]),
                rhoG=np.array([rhoG_]),
                muL=np.array([fluid.muL]),
                muG=np.array([fluid.muG]),
                CL=np.array([CL_]),
                CG=np.array([CG_]),
                sigma=np.array([fluid.sigma]),
                beta=beta,
                hD=hD,
            )
            pocket_alpha = float(np.clip(equilibrium_fractions(inp)[2][0], 0.3, 0.95))
        from mfsim.initial.recipes import blocked_ic

        ic_full, P, aG0 = blocked_ic(
            s_nodes,
            z_nodes,
            s_front=s_front,
            s_top=s_top,
            pocket_alpha=pocket_alpha,
            P_out=P_out_bar * 1e5,
            dP_margin=dP_margin_bar * 1e5,
            qL=qL,
            qG=qG,
            fluid=fluid,
            PR=PR,
        )
        if single_pipe:
            pipes[0].ic = ic_full
            if upstream_volume > 0:
                vertices[0].HG0 = float(aG0[0])
                vertices[0].PT0 = float(P[0] / PR)
        else:
            # distribui a CI pelos dutos (interpolação na coordenada curvilínea global)
            # e inicializa os vértices dinâmicos com o estado da junção
            s_off = 0.0
            for i, pipe in enumerate(pipes):
                pts = pipe.points
                m_i = pipe_mesh(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    pipe.n_elements,
                    grading=None if pipe.grading is None else list(pipe.grading),
                )
                s_glob = s_off + m_i["s"]
                pipe.ic = np.stack([np.interp(s_glob, s_nodes, ic_full[:, c]) for c in range(4)], axis=1)
                s_off += m_i["L"]
                if i < len(pipes) - 1:
                    vtx = net.vertex(pipe.v_to)
                    vtx.HG0 = float(np.interp(s_off, s_nodes, aG0))
                    vtx.PT0 = float(np.interp(s_off, s_nodes, P / PR))
        case_meta_extra = dict(pocket_alpha=pocket_alpha, s_front=s_front, s_top=s_top)
    else:
        case_meta_extra = {}
    case_meta_extra.update(
        elbow_radius=elbow_radius,
        n_elbow=n_elbow,
        s_elbow0=s_elbow0,
        grading_down=grading_down,
        grading_riser=grading_riser,
        upstream_volume=upstream_volume,
    )
    if start == "shut_in" and single_pipe:
        # estado inicial: fluido parado, fração de gás prescrita por trecho e
        # pressão hidrostática (densidade da mistura) a partir da saída
        from mfsim.geometry import pipe_mesh

        mesh = pipe_mesh(
            x, y, z, pipes[0].n_elements, grading=None if pipes[0].grading is None else list(pipes[0].grading)
        )
        s_nodes = mesh["s"]
        z_nodes = mesh["xyz"][:, 2]
        n_nodes = s_nodes.size
        from mfsim.initial.recipes import shut_in_ic

        pipes[0].ic = shut_in_ic(
            s_nodes,
            z_nodes,
            s_elbow0=s_elbow0,
            alpha_line=alpha_line0,
            alpha_riser=alpha_riser0,
            P_out=P_out_bar * 1e5,
            fluid=fluid,
            PR=PR,
        )
    net.meta = dict(
        L_h1=L_h1,
        L_down=L_down,
        slope_down_deg=slope_down_deg,
        H_riser=H_riser,
        L_h2=L_h2,
        qL=qL,
        qG=qG,
        P_out_bar=P_out_bar,
        **case_meta_extra,
    )
    return net


def generate(**kw):
    """Grafo (`Network`) de um caso de intermitência severa a partir dos parâmetros de
    `severe_slug_case` (geometria, vazões, partida, opções em `extra`), com a condição inicial
    como receita quando é duto único. É a forma de criar casos novos em JSON:

        net = generate(D=0.05, L_down=300.0, ..., start="blocked")
        mfsim.io.case_to_json(net, "cases/severe_slug/novo.json")
    """
    net = severe_slug_case(**kw)
    m = net.meta
    if kw.get("single_pipe", True) and kw.get("start", "guess") in ("blocked", "shut_in"):
        pipe = net.pipes[0]
        if kw.get("start") == "blocked":
            pipe.ic_recipe = {
                "recipe": "blocked",
                "s_front": float(m["s_front"]),
                "s_top": float(m["s_top"]),
                "pocket_alpha": float(m["pocket_alpha"]),
                "P_out_bar": float(m["P_out_bar"]),
                "dP_margin_bar": float(kw.get("dP_margin_bar", 0.3)),
                "qL": float(m["qL"]),
                "qG": float(m["qG"]),
            }
        else:
            pipe.ic_recipe = {
                "recipe": "shut_in",
                "s_elbow0": float(m["s_elbow0"]),
                "alpha_line": float(kw.get("alpha_line0", 0.5)),
                "alpha_riser": float(kw.get("alpha_riser0", 0.01)),
                "P_out_bar": float(m["P_out_bar"]),
            }
        pipe.ic = None
    net.meta = dict(
        m,
        generator="severe_slug_case",
        generator_params={k: (v if not hasattr(v, "tolist") else v.tolist()) for k, v in kw.items() if k != "fluid"},
    )
    return net


def main(argv=None):
    """`python -m generators.severe_slug --out novo.json --param D=0.05 --param start=blocked`."""
    import argparse
    import json as _json

    from mfsim.io.spec import case_to_json

    ap = argparse.ArgumentParser(
        description="Gera um caso de intermitência severa em JSON a partir dos parâmetros de severe_slug_case."
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--param",
        action="append",
        default=[],
        help="chave=valor (JSON) de severe_slug_case, ex.: --param D=0.05 --param start=blocked",
    )
    a = ap.parse_args(argv)
    kw = {}
    for item in a.param:
        k, v = item.split("=", 1)
        try:
            kw[k] = _json.loads(v)
        except _json.JSONDecodeError:
            kw[k] = v
    case_to_json(generate(**kw), a.out)
    print("escrito", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
