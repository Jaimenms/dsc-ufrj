"""Estimativa inicial para o estado estacionário de uma rede.

Para cada duto: vazões iguais às especificações de entrada (ou às do vértice
de origem), pressão a partir da pressão de saída com estimativa hidrostática,
fração de gás a partir das frações de equilíbrio do padrão fixado
(Taitel/Pots/Zuber) ou, se automático, da fração estratificada de equilíbrio.
"""

from __future__ import annotations

import numpy as np

from .closures import ClosureInputs, equilibrium_fractions
from .closures.regime import (
    REGIME_ANNULAR,
    REGIME_AUTO,
    REGIME_BUBBLE,
    REGIME_INTERMITTENT,
)
from .geometry import stratified_level
from .models.base import G, SLOT_ALPHA, SLOT_P, SLOT_QG, SLOT_QL
from .network.specs import UNSPECIFIED


def _equilibrium_alpha(model, P, qL, qG, theta):
    p = model.params
    fl = model.fluid
    rhoL, rhoG, CL, CG = fl.props(P)
    A = np.pi * p.D**2 / 4
    n = np.size(P)
    vLS = qL / rhoL / A * np.ones(n)
    vGS = qG / rhoG / A * np.ones(n)
    hD, beta = stratified_level(np.full(n, 0.5))
    inp = ClosureInputs(
        alphaG=np.full(n, 0.5), alphaL=np.full(n, 0.5), vL=vLS * 2, vLS=vLS, vG=vGS * 2, vGS=vGS,
        AreaL=np.full(n, A / 2), AreaG=np.full(n, A / 2), D=np.full(n, p.D), eps=np.full(n, p.eps),
        theta=theta * np.ones(n), rhoL=rhoL * np.ones(n), rhoG=rhoG * np.ones(n),
        muL=np.full(n, fl.muL), muG=np.full(n, fl.muG), CL=CL * np.ones(n), CG=CG * np.ones(n),
        sigma=np.full(n, fl.sigma), beta=beta, hD=hD,
    )
    hD_est, hD_anu, aG_est, aG_anu, aG_bol = equilibrium_fractions(inp)
    if p.regime == REGIME_ANNULAR:
        aG = aG_anu
    elif p.regime == REGIME_BUBBLE:
        aG = aG_bol
    elif p.regime == REGIME_INTERMITTENT:
        aG = 0.5 * (aG_bol + aG_est)
    else:
        aG = aG_est
    return np.clip(aG, 0.02, 0.98)


def initial_guess(network, t=None):
    """Vetor de estados inicial estimado para a rede `network` (NetworkModel)."""
    t = network.settings.t0 if t is None else t
    E = np.array([[tb(t) for tb in row] for row in network.tables], dtype=float)
    PR = network.PR
    X = np.zeros(network.n_states)
    # pressão de referência: alguma especificação de pressão na rede (ou P0 do fluido)
    P_specs = [E[i, SLOT_P] for i in range(network.NV) if E[i, SLOT_P] != UNSPECIFIED and E[i, SLOT_P] > 0]
    PT_ref = float(np.mean(P_specs)) if P_specs else network.fluid.P0 / PR
    # pressões nos vértices: especificadas ou propagadas de jusante para montante
    PT_vertex = {i: (E[i, SLOT_P] if (E[i, SLOT_P] != UNSPECIFIED and E[i, SLOT_P] > 0
                                      and network.vertices[i].kind == "B") else None)
                 for i in range(network.NV)}
    # vazões por duto: da especificação do vértice de origem ou herdadas de montante
    q_pipe = {}
    for _ in range(len(network.pipes) + 1):
        for pb in network.pipes:
            vin = network.vpos[pb.pipe.v_in]
            qL = E[vin, SLOT_QL] if E[vin, SLOT_QL] != UNSPECIFIED else None
            qG = E[vin, SLOT_QG] if E[vin, SLOT_QG] != UNSPECIFIED else None
            if network.vertices[vin].kind == "A" or qL is None or qG is None or (qL <= 0 and qG <= 0):
                ups = [q_pipe[k] for k in network.pipes_in[vin] if k in q_pipe]
                if ups:
                    qL, qG = (sum(u[0] for u in ups), sum(u[1] for u in ups))
                elif qL is None or qG is None or (qL <= 0 and qG <= 0):
                    qL, qG = 1.0, 0.1
            q_pipe[network.pipes.index(pb)] = (qL, qG)
    PT_pipe_in = {}
    for _ in range(len(network.pipes) + 1):
        for k in reversed(range(len(network.pipes))):
            pb = network.pipes[k]
            vout = network.vpos[pb.pipe.v_out]
            PT_out = PT_vertex.get(vout)
            if PT_out is None:
                downs = [PT_pipe_in[j] for j in network.pipes_out[vout] if j in PT_pipe_in]
                PT_out = float(np.mean(downs)) if downs else PT_ref
            qL, qG = q_pipe[k]
            theta = pb.scheme.theta_nodes
            z = np.concatenate([[0.0], np.cumsum(np.sin(pb.mesh["theta"]) * pb.mesh["h"])])
            model = pb.model
            if model.name in ("twofluid", "beggs_brill"):
                aG0 = float(np.mean(_equilibrium_alpha(model, PT_out * PR, qL, qG, float(np.mean(theta)))))
            else:
                aG0 = 1.0 if model.name == "gas" else 0.0
            rhoL, rhoG, _, _ = network.fluid.props(PT_out * PR)
            rho_mix = (1 - aG0) * rhoL + aG0 * rhoG
            PT_pipe_in[k] = (PT_out * PR + rho_mix * G * (z[-1] - z[0]) + 1.0 * pb.mesh["L"]) / PR
            pb._guess = (PT_out, aG0, qL, qG)
    for pb in network.pipes:
        vin = network.vpos[pb.pipe.v_in]
        vout = network.vpos[pb.pipe.v_out]
        PT_out, aG0, qL, qG = pb._guess
        x = pb.scheme.x_nodes
        theta = pb.scheme.theta_nodes
        z = np.concatenate([[0.0], np.cumsum(np.sin(pb.mesh["theta"]) * pb.mesh["h"])])
        z_nodes = np.interp(x, pb.mesh["s"], z)
        model = pb.model
        rhoL, rhoG, _, _ = network.fluid.props(PT_out * PR)
        rho_mix = (1 - aG0) * rhoL + aG0 * rhoG
        # hidrostática + pequena perda por atrito (1 Pa/m)
        P = PT_out * PR + rho_mix * G * (z_nodes[-1] - z_nodes) + 1.0 * (x[-1] - x)
        Y = pb.base_Y()
        Y[:, SLOT_ALPHA] = aG0 if model.name in ("twofluid", "beggs_brill") else Y[:, SLOT_ALPHA]
        Y[:, SLOT_P] = P / PR
        if SLOT_QL in model.slots:
            Y[:, SLOT_QL] = qL
        if SLOT_QG in model.slots:
            Y[:, SLOT_QG] = qG
        if hasattr(pb.scheme, "from_primitive"):
            Y = pb.scheme.from_primitive(Y)
        X[pb.offset:pb.offset + pb.n_states] = Y.ravel()[pb.free]
    for k, vi in enumerate(network.vA):
        v = network.vertices[vi]
        iH, iP = network.vertex_A_index(v.index)
        aG, PT = [], []
        for kp in network.pipes_in[vi] + network.pipes_out[vi]:
            pb = network.pipes[kp]
            seg = X[pb.offset:pb.offset + pb.n_states]
            Y = pb.base_Y().ravel(); Y[pb.free] = seg; Y = Y.reshape(pb.n_nodes, 4)
            if hasattr(pb.scheme, "to_primitive"):
                Y = pb.scheme.to_primitive(Y)
            node = pb.outlet_P_node if kp in network.pipes_in[vi] else pb.inlet_P_node
            if pb.model.name in ("twofluid", "beggs_brill"):
                aG.append(Y[node, SLOT_ALPHA])
            PT.append(Y[node, SLOT_P])
        X[iH] = float(np.mean(aG)) if aG else 0.5
        X[iP] = float(np.mean(PT)) if PT else PT_ref
    return X
