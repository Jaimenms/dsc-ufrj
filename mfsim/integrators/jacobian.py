"""Jacobiana numérica esparsa por diferenças finitas com coloração de colunas."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def color_groups(S):
    """Coloração gulosa das colunas de um padrão de esparsidade booleano S.

    Duas colunas podem partilhar a mesma cor se não tiverem linhas em comum.
    Retorna lista de arrays de índices de colunas.
    """
    S = sp.csc_matrix(S)
    n = S.shape[1]
    # grafo de conflitos: colunas i, j conflitam se (S^T S)_ij != 0
    C = (S.T @ S).tocsr()
    colors = -np.ones(n, dtype=int)
    order = np.argsort(-np.diff(C.indptr))  # maior grau primeiro
    for i in order:
        neigh = C.indices[C.indptr[i] : C.indptr[i + 1]]
        used = set(colors[neigh][colors[neigh] >= 0].tolist())
        c = 0
        while c in used:
            c += 1
        colors[i] = c
    ncol = colors.max() + 1
    return [np.flatnonzero(colors == c) for c in range(ncol)]


def colored_jacobian(fun, x, f0=None, S=None, groups=None, rel=1e-7, abs_=1e-9):
    """J = df/dx (esparsa CSC) por diferenças progressivas com coloração.

    fun(x) -> f (n,). S: padrão de esparsidade (n x n) booleano.
    """
    x = np.asarray(x, float)
    n = x.size
    if f0 is None:
        f0 = fun(x)
    if S is None:
        S = np.ones((n, n), dtype=bool)
    S = sp.csc_matrix(S)
    if groups is None:
        groups = color_groups(S)
    rows, cols, vals = [], [], []
    for g in groups:
        h = rel * np.abs(x[g]) + abs_
        xp = x.copy()
        xp[g] += h
        df = fun(xp) - f0
        for j, hj in zip(g, h):
            r = S.indices[S.indptr[j] : S.indptr[j + 1]]
            rows.append(r)
            cols.append(np.full(r.size, j))
            vals.append(df[r] / hj)
    rows = np.concatenate(rows) if rows else np.array([], int)
    cols = np.concatenate(cols) if cols else np.array([], int)
    vals = np.concatenate(vals) if vals else np.array([])
    return sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
