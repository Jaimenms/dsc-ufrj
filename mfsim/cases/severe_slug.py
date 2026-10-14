"""Caso de intermitência severa (severe slugging).

Geometria clássica (Taitel 1986; Schmidt et al. 1980): linha horizontal,
trecho descendente (inclinação negativa) que termina na base de um riser
vertical, seguido de um trecho horizontal até o separador (pressão fixa).

Critério de Bøe (1981) para ocorrência de intermitência severa em
linha-riser:  vSL >= (P / (rhoL g (1 - alphaG) L)) vSG   (Bøe, 1981; sem fator de inclinação)
onde L é o comprimento do trecho descendente, P a pressão na base do riser.
"""

from __future__ import annotations

import numpy as np

from ..fluid import Fluid
from ..network.data import BC_ABSENT, BC_DIRICHLET, BC_NEUMANN, Case, Pipe, SimulatorSettings, Vertex

G = 9.80665


def boe_criterion(vSL, vSG, P_base, rhoL, alphaG, L_gas):
    """Critério de Bøe (1981) na forma citada por Taitel (1986) e Jansen et al. (1996):

        vSL >= P / (rhoL g (1 - alphaG) L) * vSG

    Compara a taxa de subida da coluna hidrostática no riser (rhoL g vSL) com a taxa
    de compressão do gás da linha (P vSG / (alphaG L)); L é o comprimento de linha
    ocupado pelo gás (bolsão + volume a montante), sem fator de inclinação. Uma
    versão anterior desta função multiplicava L por sin|theta|, o que não consta
    das fontes e superestimava vSL_min por 1/sin|theta| (19x a -3 graus).
    Retorna (vSL_min, severe) com severe = vSL >= vSL_min."""
    vSL_min = P_base / (rhoL * G * (1.0 - alphaG) * L_gas) * vSG
    return vSL_min, vSL >= vSL_min


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
    """Constrói o caso: um único duto com 4 trechos (ou 4 dutos acoplados por
    vértices dinâmicos se single_pipe=False)."""
    fluid = fluid or Fluid(rhoL0=850.0, rhoG0=18.0, muL=5e-3, muG=1.5e-5, gammaL=1000.0, gammaG=1.1,
                           P0=20e5, sigma=0.03)
    PR = 1e6
    th = np.deg2rad(slope_down_deg)
    dy_down = L_down * np.cos(th)
    dz_down = L_down * np.sin(th)
    # traçado (y horizontal, z vertical)
    yC, zC = L_h1 + dy_down, dz_down                      # joelho (base do riser)
    if elbow_radius > 0 and n_elbow > 0:
        # joelho em arco de raio R tangente ao trecho descendente e ao riser: os
        # elementos do arco têm ângulos intermediários entre slope_down e 90°, o que
        # dá sentido à transição suave dos fechamentos por ângulo (vertical_blend)
        R = elbow_radius
        t_tan = R * (1.0 - np.sin(th)) / np.cos(th)      # distância do joelho aos pontos de tangência
        y1, z1 = yC - t_tan * np.cos(th), zC - t_tan * np.sin(th)
        ang = th + (np.pi / 2 - th) * np.arange(1, n_elbow + 1) / n_elbow
        y_arc = y1 + R * (np.sin(ang) - np.sin(th))
        z_arc = z1 - R * (np.cos(ang) - np.cos(th))
        y = [0.0, L_h1, y1] + y_arc.tolist() + [yC, yC + L_h2]
        z = [0.0, 0.0, z1] + z_arc.tolist() + [zC + H_riser, zC + H_riser]
        n_el_list = [n_h1, n_down] + [1] * n_elbow + [n_riser, n_h2]
        grading = [1.0, grading_down] + [1.0] * n_elbow + [grading_riser, 1.0]
        s_elbow0 = L_h1 + L_down - t_tan                   # início do arco (frente de líquido na CI bloqueada)
        # grupos de segmentos por duto no modo rede: linha (horizontal + descendente), pescoço (arco), riser, saída
        groups = [[0, 1], list(range(2, 2 + n_elbow)), [2 + n_elbow], [3 + n_elbow]]
    else:
        y = [0.0, L_h1, yC, yC, yC + L_h2]
        z = [0.0, 0.0, zC, zC + H_riser, zC + H_riser]
        n_el_list = [n_h1, n_down, n_riser, n_h2]
        grading = [1.0, grading_down, grading_riser, 1.0]   # malha não uniforme: < 1 refina para o joelho (descendente), > 1 refina a base (riser)
        s_elbow0 = L_h1 + L_down
        groups = [[0, 1], [2], [3]]
    x = [0.0] * len(y)
    settings = SimulatorSettings(t0=0.0, tf=tf, dt_out=dt_out, PR=PR, method=method, max_step=np.inf,
                                 atol=1e-6, rtol=1e-4, quadrature_order=2, integrator="implicit")
    if start == "shut_in":
        # partida de linha fechada: vazões sobem em rampa de 0 até o valor nominal
        spec_in = {"qL": (np.array([0.0, t_ramp]), np.array([0.0, qL])),
                   "qG": (np.array([0.0, t_ramp]), np.array([0.0, qG]))}
    else:
        spec_in = {"qL": qL, "qG": qG}
    if upstream_volume > 0:
        # volume de gás a montante (gasoduto comprimível): vértice dinâmico com as vazões de
        # alimentação como entradas externas; o duto recebe alphaG e P do vértice e as
        # vazões na face de entrada passam a ser estados (só no duto único)
        if not single_pipe:
            raise ValueError("upstream_volume só está implementado com single_pipe=True")
        vertices = [
            Vertex(index=1, kind="A", volume=upstream_volume, spec=spec_in),
            Vertex(index=2, kind="B", spec={"PT": P_out_bar * 1e5 / PR}),
        ]
    else:
        vertices = [
            Vertex(index=1, kind="B", spec=spec_in),
            Vertex(index=2, kind="B", spec={"PT": P_out_bar * 1e5 / PR}),
        ]
    extra = dict(extra or {})
    if model == 2:
        # modelo de deriva no escalonado: o hold-up da condição inicial é o dado (bolsão,
        # riser cheio), não o da correlação com as vazões nominais
        extra.setdefault("ic_holdup", "given")
    if single_pipe:
        bc_in_single = (BC_DIRICHLET, BC_DIRICHLET, BC_ABSENT, BC_ABSENT) if upstream_volume > 0 else \
                       (BC_NEUMANN, BC_ABSENT, BC_DIRICHLET, BC_DIRICHLET)
        pipe = Pipe(index=1, v_in=1, v_out=2, x=x, y=y, z=z, n_elements=n_el_list,
                    D=D, roughness=roughness, model=model, regime=regime,
                    bc_in=bc_in_single,
                    bc_out=(BC_ABSENT, BC_DIRICHLET, BC_ABSENT, BC_ABSENT), ic=None, extra=extra, grading=grading)
        pipes = [pipe]
    else:
        # rede: um duto por grupo de trechos (linha, pescoço, riser, saída), acoplados
        # por vértices dinâmicos (balanço de massa num volume, sem momento); as
        # condições de contorno internas são alphaG e P impostos pelo vértice na
        # entrada do duto a jusante e P imposto pelo vértice na saída do duto a montante
        n_pipes = len(groups)
        regimes = list(regime) if isinstance(regime, (list, tuple)) else [regime] * n_pipes
        vol = extra.pop("vertex_volume", 0.05)
        junctions = [3 + j for j in range(n_pipes - 1)]
        vertices = [vertices[0]] + [Vertex(index=j, kind="A", volume=vol, spec={"qL": 0.0, "qG": 0.0}) for j in junctions] + [vertices[1]]
        vs = [1] + junctions + [2]
        pipes = []
        for i, g in enumerate(groups):
            seg_y = [y[g[0]]] + [y[k + 1] for k in g]
            seg_z = [z[g[0]]] + [z[k + 1] for k in g]
            bc_in = (BC_NEUMANN, BC_ABSENT, BC_DIRICHLET, BC_DIRICHLET) if i == 0 else \
                    (BC_DIRICHLET, BC_DIRICHLET, BC_ABSENT, BC_ABSENT)
            # saída para vértice dinâmico: alphaG do vértice no fantasma (célula doadora em refluxo);
            # saída para o separador: alphaG livre
            bc_out = (BC_DIRICHLET, BC_DIRICHLET, BC_ABSENT, BC_ABSENT) if i < n_pipes - 1 else \
                     (BC_ABSENT, BC_DIRICHLET, BC_ABSENT, BC_ABSENT)
            pipes.append(Pipe(index=i + 1, v_in=vs[i], v_out=vs[i + 1], x=[0.0] * len(seg_y), y=seg_y, z=seg_z,
                              n_elements=[n_el_list[k] for k in g], D=D, roughness=roughness, model=model,
                              regime=regimes[i], bc_in=bc_in, bc_out=bc_out,
                              ic=None, extra=dict(extra), grading=[grading[k] for k in g]))
    case = Case(name="severe_slug", fluid=fluid, settings=settings, vertices=vertices, pipes=pipes)
    if start == "blocked":
        from ..geometry import pipe_mesh
        mesh = pipe_mesh(x, y, z, n_el_list, grading=grading)
        s_nodes = mesh["s"]
        z_nodes = mesh["xyz"][:, 2]
        n_nodes = s_nodes.size
        s_riser0 = s_elbow0                      # ponto baixo: início do arco (ou o joelho)
        s_top = mesh["L"] - L_h2                 # topo do riser
        s_front = s_riser0 - front_liquid_length
        if pocket_alpha is None:
            # fração de gás de equilíbrio do estratificado no trecho descendente
            from ..closures import ClosureInputs, equilibrium_fractions
            from ..geometry import stratified_level
            P_est = P_out_bar * 1e5 + fluid.rhoL0 * G * H_riser
            rhoL_, rhoG_, CL_, CG_ = fluid.props(P_est)
            Area = np.pi * D**2 / 4
            vLS = qL / rhoL_ / Area
            vGS = qG / rhoG_ / Area
            hD, beta = stratified_level(np.array([0.5]))
            inp = ClosureInputs(alphaG=np.array([0.5]), alphaL=np.array([0.5]), vL=np.array([2 * vLS]), vLS=np.array([vLS]),
                                vG=np.array([2 * vGS]), vGS=np.array([vGS]), AreaL=np.array([Area / 2]), AreaG=np.array([Area / 2]),
                                D=np.array([D]), eps=np.array([roughness]), theta=np.array([th]), rhoL=np.array([rhoL_]),
                                rhoG=np.array([rhoG_]), muL=np.array([fluid.muL]), muG=np.array([fluid.muG]),
                                CL=np.array([CL_]), CG=np.array([CG_]), sigma=np.array([fluid.sigma]), beta=beta, hD=hD)
            pocket_alpha = float(np.clip(equilibrium_fractions(inp)[2][0], 0.3, 0.95))
        aG0 = np.where(s_nodes < s_front - 1e-9, pocket_alpha, 0.0)
        aG0 = np.where(s_nodes > s_top + 1e-9, 0.0, aG0)   # horizontal de saída
        P = np.empty(n_nodes)
        P[-1] = P_out_bar * 1e5
        for i in range(n_nodes - 2, -1, -1):
            rhoL_, rhoG_, _, _ = fluid.props(P[i + 1])
            if s_nodes[i] < s_front - 1e-9:
                rho_h = rhoG_          # no bolsão estratificado a pressão (do gás) é quase uniforme
            else:
                a = 0.5 * (aG0[i] + aG0[i + 1])
                rho_h = (1 - a) * rhoL_ + a * rhoG_
            P[i] = P[i + 1] + rho_h * G * (z_nodes[i + 1] - z_nodes[i])
            if s_nodes[i] < s_front - 1e-9 <= s_nodes[i + 1]:
                P[i] += dP_margin_bar * 1e5          # salto de pressão na frente de líquido
        qG0 = np.where(s_nodes < s_front - 1e-9, qG, 0.0)
        ic_full = np.stack([aG0, P / PR, np.full(n_nodes, qL), qG0], axis=1)
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
                m_i = pipe_mesh(pipe.x, pipe.y, pipe.z, pipe.n_elements, grading=pipe.grading)
                s_glob = s_off + m_i["s"]
                pipe.ic = np.stack([np.interp(s_glob, s_nodes, ic_full[:, c]) for c in range(4)], axis=1)
                s_off += m_i["L"]
                if i < len(pipes) - 1:
                    vtx = next(v for v in vertices if v.index == pipe.v_out)
                    vtx.HG0 = float(np.interp(s_off, s_nodes, aG0))
                    vtx.PT0 = float(np.interp(s_off, s_nodes, P / PR))
        case_meta_extra = dict(pocket_alpha=pocket_alpha, s_front=s_front)
    else:
        case_meta_extra = {}
    case_meta_extra.update(elbow_radius=elbow_radius, n_elbow=n_elbow, s_elbow0=s_elbow0, grading_down=grading_down, grading_riser=grading_riser,
                           upstream_volume=upstream_volume)
    if start == "shut_in" and single_pipe:
        # estado inicial: fluido parado, fração de gás prescrita por trecho e
        # pressão hidrostática (densidade da mistura) a partir da saída
        from ..geometry import pipe_mesh
        mesh = pipe_mesh(x, y, z, pipes[0].n_elements, grading=pipes[0].grading)
        s_nodes = mesh["s"]
        z_nodes = mesh["xyz"][:, 2]
        n_nodes = s_nodes.size
        s_riser0 = s_elbow0
        aG0 = np.where(s_nodes < s_riser0 - 1e-9, alpha_line0, alpha_riser0)
        P = np.empty(n_nodes)
        P[-1] = P_out_bar * 1e5
        for i in range(n_nodes - 2, -1, -1):
            rhoL_, rhoG_, _, _ = fluid.props(P[i + 1])
            a = 0.5 * (aG0[i] + aG0[i + 1])
            rho_mix = (1 - a) * rhoL_ + a * rhoG_
            P[i] = P[i + 1] + rho_mix * G * (z_nodes[i + 1] - z_nodes[i])
        pipes[0].ic = np.stack([aG0, P / PR, np.zeros(n_nodes), np.zeros(n_nodes)], axis=1)
    case.meta = dict(L_h1=L_h1, L_down=L_down, slope_down_deg=slope_down_deg, H_riser=H_riser, L_h2=L_h2,
                     qL=qL, qG=qG, P_out_bar=P_out_bar, **case_meta_extra)
    # faixas esperadas dos estados (para tolerâncias absolutas uniformes, tol_range):
    # os picos de blowdown chegam a ~20x a alimentação de líquido e ~50x a de gás
    case.settings.extra.setdefault("state_ref", dict(
        qL=20.0 * qL, qG=50.0 * qG,
        PT=(P_out_bar * 1e5 + 1.1 * fluid.rhoL0 * G * H_riser + 2e5) / PR))
    return case


def superficial_velocities(case):
    p = case.pipes[0]
    A = np.pi * p.D**2 / 4
    fl = case.fluid
    P = case.meta["P_out_bar"] * 1e5
    rhoL, rhoG, _, _ = fl.props(P)
    return case.meta["qL"] / rhoL / A, case.meta["qG"] / rhoG / A
