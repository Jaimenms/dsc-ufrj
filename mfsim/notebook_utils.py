"""Funções de apoio aos notebooks de reprodução do capítulo 10."""

from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt

from .cases import CHAPTER10, load_chapter10_case
from .io_legacy import legacy_states
from .plotting import legacy_pipe_dict, pipe_surfaces, pipe_timeseries
from .simulation import build_model, simulate


def describe(key):
    info = CHAPTER10[key]
    print(f"{info.section} {info.title}\n{info.description}\nFiguras da tese: {info.figures}")
    return info


def run_case(key, integrator="implicit", rtol=1e-4, atol=1e-6, method=None, ic="legacy", **kw):
    """Carrega o caso, parte do estado inicial do MATLAB e simula.

    Retorna (case, raw, res). O estado inicial é o gravado no .mf2 (ic="legacy")
    para que a comparação com o MATLAB seja direta.
    """
    case, raw = load_chapter10_case(key)
    if method is not None:
        case.settings.method = method
    t0 = time.time()
    res = simulate(case, integrator=integrator, rtol=rtol, atol=atol, max_step=np.inf, progress=True, **kw)
    res.cpu = time.time() - t0
    print(f"tempo de CPU: {res.cpu:.2f} s")
    return case, raw, res


def compare_states(res, raw, label="MATLAB"):
    """Tabela de erro máximo por variável e por duto em relação aos estados gravados."""
    Tl, Xl = legacy_states(raw)
    model = res.model
    if Xl is None or Xl.shape != res.X.shape:
        print("estados gravados incompatíveis para comparação direta")
        return
    rows = []
    for k, pb in enumerate(model.pipes):
        slots = pb.free % 4
        seg = slice(pb.offset, pb.offset + pb.n_states)
        for s, name in enumerate(["alphaG", "PT", "qL", "qG"]):
            m = slots == s
            if m.any():
                d = np.max(np.abs(res.X[seg][m] - Xl[seg][m]))
                rng = np.ptp(Xl[seg][m])
                rows.append((k + 1, pb.model.name, name, d, rng))
    print(f"{'duto':>4s} {'modelo':12s} {'var':7s} {'máx |dif|':>10s} {'faixa ' + label:>14s} {'dif/faixa':>10s}")
    for k, mn, name, d, rng in rows:
        print(f"{k:4d} {mn:12s} {name:7s} {d:10.2e} {rng:14.2e} {d / rng if rng > 0 else 0:10.2e}")
    if model.vA:
        seg = slice(model.n_pipe_states, None)
        print("vértices dinâmicos: máx |dif| =", np.max(np.abs(res.X[seg] - Xl[seg]), axis=1))


def show_pipe(res, raw, k=0, title=None, nodes=None):
    legacy = legacy_pipe_dict(raw, k)
    fig = pipe_surfaces(res, k, title=title)
    plt.show()
    fig = pipe_timeseries(res, k, legacy=legacy, nodes=nodes)
    fig.suptitle(f"Tubulação {k + 1}: Python (cheio) × MATLAB (tracejado)", y=1.02)
    plt.show()


def plot_elements(res, raw=None):
    """Séries temporais de válvulas, reservatórios e válvulas de retenção."""
    el = res.elements
    leg = raw.get("saidas", {}) if raw else {}
    for kind, lkey in (("valves", "valvula"), ("reservoirs", "reservatorio"), ("check_valves", "valvulaRet")):
        for i, (v, d) in enumerate(el[kind].items()):
            keys = [kk for kk in d if kk not in ("P_down",)]
            fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3))
            lg = leg.get(lkey)
            if isinstance(lg, dict):
                lg = [lg]
            for ax, kk in zip(np.atleast_1d(axes), keys):
                ax.plot(res.T, d[kk], "b-", label="Python")
                if lg is not None and i < len(lg) and kk in lg[i]:
                    T = np.asarray(raw["estados"]["T"]).ravel()
                    ax.plot(T, np.asarray(lg[i][kk], float).ravel(), "r--", label="MATLAB")
                ax.set_title(f"{kind[:-1]} (vértice {v}): {kk}", fontsize=9)
                ax.set_xlabel("Tempo (s)")
                ax.grid(alpha=0.3)
            np.atleast_1d(axes)[0].legend(fontsize=7)
            fig.tight_layout()
            plt.show()


def plot_vertices(res, raw=None, indices=None):
    model = res.model
    indices = indices or [v.id for v in model.vertices if v.kind != "junction"]
    leg = raw["saidas"]["vertice"] if raw else None
    T = np.asarray(raw["estados"]["T"]).ravel() if raw else None
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for idx in indices:
        v = res.vertex(idx)
        for ax, (kk, lk) in zip(axes, (("pressao", "pressao"), ("vazaoMassaL", "vazaoMassaL"), ("vazaoMassaG", "vazaoMassaG"))):
            line, = ax.plot(res.T, v[kk], label=f"vértice {idx}")
            if leg is not None:
                lv = leg[model.vpos[idx]]
                if lk in lv:
                    ax.plot(T, np.asarray(lv[lk], float).ravel(), "--", color=line.get_color(), alpha=0.7)
    for ax, kk in zip(axes, ("Pressão (bar)", "Vazão de líquido (kg/s)", "Vazão de gás (kg/s)")):
        ax.set_ylabel(kk); ax.set_xlabel("Tempo (s)"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    plt.show()


def jax_comparison(case, res_numpy=None, label="", **simulate_kw):
    """Repete a simulação do caso com o lado direito compilado (JAX) e compara com a
    versão numpy: tempo de CPU (sem a compilação, que é feita uma vez e guardada em
    cache no disco), passos e diferença máxima relativa dos estados. Devolve o Results
    da versão JAX."""
    from .compiled import get_jax
    from .simulation import build_model
    model = build_model(case)
    t0 = time.time()
    rhs_j, jac_j = get_jax(model)
    x0 = model.initial_state()
    rhs_j(case.settings.t0, x0); jac_j(case.settings.t0, x0)
    t_compile = time.time() - t0
    kw = dict(integrator="implicit", max_step=np.inf, progress=False)
    kw.update(simulate_kw)
    t0 = time.time()
    res_j = simulate(case, jax=True, **kw)
    t_jax = time.time() - t0
    msg = f"JAX{(' ' + label) if label else ''}: {t_jax:.2f} s ({res_j.info['n_steps']} passos; compilação {t_compile:.2f} s, uma vez por modelo)"
    if res_numpy is not None:
        cpu_np = getattr(res_numpy, "cpu", res_numpy.info.get("cpu", np.nan))
        n = min(res_numpy.X.shape[1], res_j.X.shape[1])
        scale = np.max(np.abs(res_numpy.X), axis=1) + 1e-12
        d = np.max(np.abs(res_numpy.X[:, :n] - res_j.X[:, :n]) / scale[:, None])
        msg += f" | numpy: {cpu_np:.2f} s ({res_numpy.info['n_steps']} passos) | diferença máxima relativa dos estados {d:.1e}"
    print(msg)
    return res_j
