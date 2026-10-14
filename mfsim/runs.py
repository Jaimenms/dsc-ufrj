"""Corridas gravadas (formato v2): `T` (nt,), `X` (n_states, nt), opcional `acc` (integral
BDF-consistente dos fluxos de contorno), `case_json` (o caso completo em `mfsim-case/1`,
com as condições iniciais explícitas em `ic/<duto>`), `info_json` (passos, rejeições, CPU,
sucesso/erro) e `provenance_json` (commit, versões, comando).

`load_run(nome)` procura em `runs/`, `notebooks/data/` e no arquivo; `Run` reconstrói
caso, modelo e resultados a partir do `case_json`. Os ficheiros no formato antigo (`meta`
v1) convertem-se com `mfsim convert-runs` (`mfsim.io.legacy_runs`).
"""

from __future__ import annotations

import copy
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np

from .simulation import build_model

RUN_DIRS = ("runs", os.path.join("notebooks", "data"), os.path.join("archive", "scratch_2026-10", "npz"))
INFO_KEYS = ("n_steps", "n_rejected", "cpu", "n_rhs", "n_jac", "success", "error", "note")


@dataclass
class Run:
    """Trajetória gravada + caso reconstruído (modelo e resultados preguiçosos)."""

    path: str
    T: np.ndarray
    X: np.ndarray
    case_json: dict
    acc: np.ndarray | None = None
    info: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    ic_arrays: dict = field(default_factory=dict)
    _case: object = None
    _model: object = None
    _res: object = None

    @property
    def case(self):
        """O grafo da corrida (`Network`), com `tf` no último instante gravado."""
        if self._case is None:
            from .io.spec import case_from_json

            self._case = case_from_json(self.case_json, ic_arrays=self.ic_arrays)
            if self.T.size:
                self._case.set_time(tf=float(self.T[-1]))
        return self._case

    def network(self, **time_kw):
        """Cópia do grafo com o tempo alterado (ex.: `run.network(tf=150.0, dt_out=2.0)`)."""
        net = copy.deepcopy(self.case)
        return net.set_time(**time_kw) if time_kw else net

    @property
    def meta(self):
        """Parâmetros da corrida: os do gerador (`meta.generator_params` do caso, no dialeto
        das corridas antigas: vSL, vSG, n_down, ...), ou o `meta` do caso, mais os contadores
        do integrador (n_steps, n_rejected, cpu)."""
        m = dict(self.case_json.get("meta") or {})
        base = dict(m.get("generator_params") or m)
        base.update({k: self.info[k] for k in ("n_steps", "n_rejected", "cpu") if k in self.info})
        return base

    @property
    def model(self):
        if self._model is None:
            self._model = build_model(self.case)
        return self._model

    @property
    def res(self):
        if self._res is None:
            from .results import Results

            self._res = Results(self.model, self.T, self.X)
        return self._res

    def pipe(self, k=0):
        return self.res.pipe(k)

    @property
    def p_last(self):
        """Saídas do duto que termina no separador (o último, em rede)."""
        return self.res.pipe(len(self.model.pipes) - 1)

    @property
    def ib(self):
        """Célula no início do joelho (base do riser) do primeiro duto."""
        p = self.pipe(0)
        return int(np.argmin(np.abs(p["x"] - self.case.meta["s_elbow0"])))

    def aG_riser_base(self):
        if self.meta.get("network", False):
            return self.res.pipe(2)["out"]["alphaG"][1]
        return self.pipe(0)["out"]["alphaG"][self.ib + self.meta.get("n_elbow", 0) + 1]

    def balance(self, phase="L", k=0):
        from .analysis import mass_balance

        return mass_balance(self.res, k, phase)

    def signature(self, min_gap=200.0, t_skip=0.0):
        from .analysis import cycle_signature

        P_in = self.pipe(0)["out"]["P_bar"][1]
        q_out = self.p_last["out"]["qL"][-1]
        return cycle_signature(self.T, P_in, q_out, min_gap=min_gap, t_skip=t_skip)


def find_run(name) -> str:
    """Caminho de uma corrida: o próprio caminho, ou `<nome>.npz` / `severe_D_<nome>.npz` em
    `runs/`, `notebooks/data/` e `archive/scratch_2026-10/npz/`."""
    name = str(name)
    if os.path.exists(name):
        return name
    base = name[:-4] if name.endswith(".npz") else name
    for d in RUN_DIRS:
        for fn in (f"{base}.npz", f"severe_D_{base}.npz"):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"corrida não encontrada: {name} (procurado em {RUN_DIRS})")


def load_run(path) -> Run:
    """Lê uma corrida v2 (caminho, ou nome procurado por `find_run`)."""
    path = find_run(path)
    with np.load(path, allow_pickle=False) as d:
        if "case_json" not in d.files:
            raise ValueError(f"{path}: corrida no formato antigo (meta v1); converta com `mfsim convert-runs {path}`")
        info = json.loads(str(d["info_json"])) if "info_json" in d.files else {}
        prov = json.loads(str(d["provenance_json"])) if "provenance_json" in d.files else {}
        case_json = json.loads(str(d["case_json"]))
        ic_arrays = {k[3:]: np.asarray(d[k], float) for k in d.files if k.startswith("ic/")}
        return Run(
            path=str(path),
            T=np.asarray(d["T"], float),
            X=np.asarray(d["X"], float),
            case_json=case_json,
            acc=np.asarray(d["acc"]) if "acc" in d.files else None,
            info=info,
            provenance=prov,
            ic_arrays=ic_arrays,
        )


def provenance(command=None) -> dict:
    """Proveniência de uma corrida: commit git, versões e plataforma."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        sha = ""
    import numpy
    import scipy

    try:
        import jax

        jax_v = jax.__version__
    except ImportError:
        jax_v = None
    return dict(
        git=sha,
        python=sys.version.split()[0],
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        jax=jax_v,
        platform=platform.platform(),
        command=command or " ".join(sys.argv),
    )


def save_run(path, T, X, case, info=None, acc=None, command=None):
    """Grava uma corrida v2: `T`, `X`, `acc` (opcional), `case_json` (o grafo completo; `Case`
    legado é convertido) com as condições iniciais explícitas em `ic/<duto>`, `info_json`
    e `provenance_json`."""
    from .io.spec import network_to_json
    from .network.graph import as_network

    extra = {}
    if acc is not None:
        extra["acc"] = np.asarray(acc)
    info = {
        k: (
            float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int, bool)) else v
        )
        for k, v in (info or {}).items()
        if k in INFO_KEYS
    }
    d, ic_store = network_to_json(as_network(case))
    for k, v in ic_store.items():
        extra[f"ic/{k}"] = np.asarray(v, float)
    np.savez(
        path,
        T=np.asarray(T, float),
        X=np.asarray(X, float),
        case_json=json.dumps(d),
        info_json=json.dumps(info),
        provenance_json=json.dumps(provenance(command)),
        **extra,
    )
    return path
