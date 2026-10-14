"""Resultados de simulação: estados, saídas por duto/vértice/elemento."""

from __future__ import annotations

import numpy as np


class Results:
    def __init__(self, model, T, X, info=None):
        self.model = model
        self.T = np.asarray(T, float)
        self.X = np.asarray(X, float)
        self.info = info or {}
        self._pipes = None
        self._vertices = None
        self._elements = None

    # --- pós-processamento ---------------------------------------------------
    def _compute(self):
        model = self.model
        nt = self.T.size
        pipes = []
        for pb in model.pipes:
            pipes.append({"x": pb.scheme.x_nodes, "theta": pb.scheme.theta_nodes, "Y": [], "out": {}})
        # saídas só dos vértices de fronteira e dos tanques (as junções são internas às arestas)
        shown = [vi for vi in range(model.NV) if model.vertices[vi].kind != "junction"]
        vert = [dict(holdupG=[], pressao=[], vazaoMassaL=[], vazaoMassaG=[]) for _ in shown]
        vidx = {}
        for k, vi in enumerate(shown):
            vid = model.vertices[vi].id
            vidx[vid] = k
            if vid.lstrip("-").isdigit():
                vidx.setdefault(int(vid), k)
        elems = {"valves": {}, "reservoirs": {}, "check_valves": {}}
        for it in range(nt):
            E, Ys, info, X = model.assemble(self.T[it], self.X[:, it])
            for k, (pb, Y) in enumerate(zip(model.pipes, Ys)):
                if hasattr(pb.scheme, "to_primitive"):
                    Y = pb.scheme.to_primitive(Y)
                pipes[k]["Y"].append(Y)
            for k_v, vi in enumerate(shown):
                QL = QG = 0.0
                for kp in model.arriving[vi]:
                    QL += Ys[kp][model.pipes[kp].outlet_q_node, 2]
                    QG += Ys[kp][model.pipes[kp].outlet_q_node, 3]
                for kp in model.leaving[vi]:
                    QL -= Ys[kp][model.pipes[kp].inlet_q_node, 2]
                    QG -= Ys[kp][model.pipes[kp].inlet_q_node, 3]
                row = model.report_row[vi]  # junção que desdobra o vértice (choke/reservatório), se houver
                vert[k_v]["holdupG"].append(E[row, 0])
                vert[k_v]["pressao"].append(E[row, 1] * model.PR / 1e5)
                vert[k_v]["vazaoMassaL"].append(QL)
                vert[k_v]["vazaoMassaG"].append(QG)
            for key in ("valves", "reservoirs", "check_valves"):
                for d in info[key]:
                    store = elems[key].setdefault(d["vertex"], {})
                    for kk, vv in d.items():
                        if kk == "vertex":
                            continue
                        store.setdefault(kk, []).append(vv)
        for k, pb in enumerate(model.pipes):
            Yarr = np.array(pipes[k]["Y"])  # (nt, n_nodes, 4)
            pipes[k]["Y"] = Yarr
            # saídas secundárias por instante
            outs = {}
            for it in range(nt):
                o = (
                    pb.scheme.model.coefficients(Yarr[it], pb.scheme.theta_nodes, outputs=True)[1]
                    if hasattr(pb.scheme, "to_primitive")
                    else pb.scheme.outputs(Yarr[it])
                )
                for kk, vv in o.items():
                    outs.setdefault(kk, []).append(np.broadcast_to(vv, (pb.n_nodes,)))
            pipes[k]["out"] = {kk: np.array(vv).T for kk, vv in outs.items()}  # (n_nodes, nt)
            pipes[k]["out"]["alphaG"] = Yarr[:, :, 0].T
            pipes[k]["out"]["PT"] = Yarr[:, :, 1].T
            pipes[k]["out"]["qL"] = Yarr[:, :, 2].T
            pipes[k]["out"]["qG"] = Yarr[:, :, 3].T
            pipes[k]["out"]["P_bar"] = Yarr[:, :, 1].T * model.PR / 1e5
        for k_v in range(len(vert)):
            vert[k_v] = {kk: np.array(vv) for kk, vv in vert[k_v].items()}
        self._vertex_index = vidx
        for key in elems:
            for v in elems[key]:
                elems[key][v] = {kk: np.array(vv) for kk, vv in elems[key][v].items()}
        self._pipes, self._vertices, self._elements = pipes, vert, elems

    @property
    def pipes(self):
        if self._pipes is None:
            self._compute()
        return self._pipes

    @property
    def vertices(self):
        if self._vertices is None:
            self._compute()
        return self._vertices

    @property
    def elements(self):
        if self._elements is None:
            self._compute()
        return self._elements

    def pipe(self, k):
        """Saídas do duto k (0-based): dict com x, theta, Y (nt x n x 4) e out."""
        return self.pipes[k]

    def vertex(self, index):
        """Saídas do vértice com o id dado (ou índice legado inteiro)."""
        return self.vertices[self._vertex_index[index]]

    def final_state(self):
        return self.X[:, -1]
