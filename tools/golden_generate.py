"""Gera os ficheiros de referência ("golden") de tests/golden com o código ATUAL.

É o único escritor de tests/golden. Cada grupo é um npz com arrays nomeados; HASHES.json
guarda o sha256 dos bytes de cada array (nível A = bit a bit), policy.json o nível de
comparação por array ("A" ou {"tier": "B", "rtol": ...}) e CHANGELOG.md o registo de
cada regeneração (data, commit, grupo, motivo, desvio máximo em relação ao golden
anterior).

uso: uv run python tools/golden_generate.py --all
     uv run python tools/golden_generate.py --only chapter10 severe
     uv run python tools/golden_generate.py --rebaseline --only severe --note "fase 2c: fechamentos no backend; desvio máx 3e-12"
grupos: chapter10, variants, severe, results, runs_meta, nightly
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import re
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
DATA = ROOT / "notebooks" / "data"
sys.path.insert(0, str(ROOT))

GROUPS = ("network", "chapter10", "variants", "severe", "results", "runs_meta", "notebook_text", "nightly")
NB_KW = {
    "10.3.2a": dict(rtol=1e-3, atol=1e-5),
    "10.3.2b": dict(rtol=1e-3, atol=1e-5),
}  # kwargs dos notebooks em run_case


def array_hash(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest()


def git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT, timeout=5
        ).stdout.strip()
    except Exception:
        return ""


def provenance():
    import numpy, scipy

    try:
        import jax

        jv = jax.__version__
    except ImportError:
        jv = None
    return dict(
        git=git_sha(),
        date=_dt.date.today().isoformat(),
        python=sys.version.split()[0],
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        jax=jv,
    )


# ----------------------------------------------------------------------------- grupos
def gen_chapter10():
    """Trajetórias completas dos casos do capítulo 10 como os notebooks as correm."""
    from mfsim.cases import chapter10_keys
    from mfsim.notebook_utils import run_case

    out = {}
    for key in chapter10_keys():
        t0 = time.time()
        case, raw, res = run_case(key, **NB_KW.get(key, {}))
        out[f"{key}/T"] = res.T
        out[f"{key}/X"] = res.X
        out[f"{key}/n_steps"] = np.array(res.info.get("n_steps", -1))
        print(f"  {key}: {res.X.shape} {res.info.get('n_steps')} passos {time.time()-t0:.1f} s")
    return {"chapter10.npz": out}


def gen_variants():
    """Cálculos secundários dos notebooks: estacionário automático, estudo de esquemas, 10.1.4
    com estacionário consistente, 10.3.1 em volumes finitos (modelo de deriva)."""
    from mfsim.cases import chapter10_case, legacy_results
    from mfsim.network.graph import ports
    from mfsim.simulation import simulate, steady_state

    out = {}
    case, raw = chapter10_case("10.1.1"), legacy_results("10.1.1")
    case_ss = copy.deepcopy(case)
    case_ss.pipes[0].ports_in = ports((3, 1, 2, 2))
    case_ss.pipes[0].ic = None
    model_ss, X_ss, info_ss = steady_state(case_ss, X0="auto", verbose=False)
    out["10.1.1_steady_auto/X"] = X_ss
    out["10.1.1_steady_auto/residual"] = np.array(info_ss["residual"])
    res2 = simulate(
        copy.deepcopy(case_ss), integrator="implicit", rtol=1e-4, atol=1e-6, max_step=np.inf, progress=False
    )
    out["10.1.1_c2/T"] = res2.T
    out["10.1.1_c2/X"] = res2.X
    for method in ("galerkin", "fd_central"):  # estudo de malha do notebook (métodos e malhas 5 e 25)
        for nel in (5, 25):
            c = copy.deepcopy(case_ss)
            c.pipes[0].n_elements = np.array([nel])
            c.pipes[0].ic = None
            c.set_method(method)
            steady_state(c, X0="auto", verbose=False)
            r = simulate(c, integrator="implicit", rtol=1e-4, atol=1e-6, max_step=np.inf, progress=False)
            out[f"10.1.1_mesh{nel}_{method}/T"] = r.T
            out[f"10.1.1_mesh{nel}_{method}/X"] = r.X
    case, raw = chapter10_case("10.1.4"), legacy_results("10.1.4")
    case_ss = copy.deepcopy(case)
    n = case_ss.pipes[0].n_elements[0] + 1
    case_ss.pipes[0].ic = np.stack([np.full(n, 0.3), np.full(n, 1.0), np.full(n, 0.5), np.full(n, 0.1)], axis=1)
    steady_state(case_ss, verbose=False)
    r = simulate(case_ss, integrator="implicit", rtol=1e-4, atol=1e-6, max_step=np.inf, progress=False)
    out["10.1.4_steady/T"] = r.T
    out["10.1.4_steady/X"] = r.X
    case, raw = chapter10_case("10.3.1"), legacy_results("10.3.1")
    case_fv = copy.deepcopy(case)
    case_fv.set_method("fv_upwind").set_options(conservative=True, q_switch=0.05)
    steady_state(case_fv, verbose=False, update_case=True)
    r = simulate(case_fv, progress=False, integrator="implicit", rtol=1e-3, atol=1e-6)
    out["10.3.1_fv_drift/T"] = r.T
    out["10.3.1_fv_drift/X"] = r.X
    out["10.3.1_fv_drift/n_steps"] = np.array(r.info.get("n_steps", -1))
    print("  variantes:", len(out), "arrays")
    return {"variants.npz": out}


REFERENCE_RUN = (
    "severe_slug_blocked_conservative_arc_tol_hv_vm_lv_bd_phys"  # config recomendada + arrasto físico, gammaL 100
)
LAB_RUN = "severe_slug_lab_jansen_tese_bloqueada"


def _severe_case(name, tf, dt_out):
    from mfsim.runs import load_run

    run = load_run(str(DATA / f"{name}.npz"))
    return run, run.network(tf=tf, dt_out=dt_out)


def gen_severe():
    """Severe slug de referência (150 s, numpy e JAX), rhs/Jacobiana/transformação nos estados
    históricos da corrida de referência e caso de laboratório (Jansen, 10 s, JAX)."""
    from mfsim.compiled import build_jax_rhs
    from mfsim.simulation import simulate

    out = {}
    run, case = _severe_case(REFERENCE_RUN, tf=150.0, dt_out=2.0)
    for backend in ("numpy", "jax"):
        t0 = time.time()
        res = simulate(case, backend=backend, progress=False)
        model, T, X, info = res.model, res.T, res.X, res.info
        out[f"reference_{backend}_150s/T"] = T
        out[f"reference_{backend}_150s/X"] = X
        out[f"reference_{backend}_150s/acc"] = info["accum"]
        out[f"reference_{backend}_150s/n_steps"] = np.array(info["n_steps"])
        print(f"  referência {backend}: {info['n_steps']} passos {time.time()-t0:.1f} s")
    # amostras nos estados históricos (a trajetória guardada não é reproduzível; o rhs nela é)
    model = run.model
    T, X = run.T, run.X
    idx = np.unique(np.linspace(0, T.size - 1, 20).astype(int))
    out["reference_samples/idx"] = idx
    out["reference_samples/rhs"] = np.stack([model.rhs(T[i], X[:, i]) for i in idx])
    out["reference_samples/transform"] = np.stack([model.transform(X[:, i]) for i in idx])
    out["reference_samples/inverse"] = np.stack([model.inverse_transform(model.transform(X[:, i])) for i in idx])
    out["reference_samples/transform_jac"] = np.stack([model.transform_jacobian(X[:, i]).toarray() for i in idx[:5]])
    out["reference_samples/state_scales"] = model.state_scales(X[:, 0])
    out["reference_samples/boundary_net_fluxes"] = np.stack(
        [np.asarray(model.boundary_net_fluxes(T[i], X[:, i]), float) for i in idx]
    )
    out["reference_samples/sparsity"] = np.asarray(model.jac_sparsity(), dtype=float).astype(np.int8)
    rhs_j, jac_j = build_jax_rhs(model)
    out["reference_samples/rhs_jax"] = np.stack([rhs_j(T[i], X[:, i]) for i in idx])
    out["reference_samples/jac_jax"] = np.stack([np.asarray(jac_j(T[i], X[:, i])) for i in idx[:3]])
    run_l, case_l = _severe_case(LAB_RUN, tf=10.0, dt_out=0.25)
    t0 = time.time()
    res = simulate(case_l, backend="jax", progress=False)
    model, T, X, info = res.model, res.T, res.X, res.info
    out["lab_jansen_jax_10s/T"] = T
    out["lab_jansen_jax_10s/X"] = X
    out["lab_jansen_jax_10s/n_steps"] = np.array(info["n_steps"])
    print(f"  Jansen 10 s JAX: {info['n_steps']} passos {time.time()-t0:.1f} s")
    return {"severe.npz": out}


def _results_arrays(prefix, res, idx):
    """Todas as saídas de Results em alguns instantes, como arrays nomeados."""
    from mfsim.results import Results

    out = {}
    model = res.model
    r = Results(model, res.T[idx], res.X[:, idx])
    for k in range(len(model.pipes)):
        p = r.pipe(k)
        out[f"{prefix}/pipe{k}/Y"] = np.asarray(p["Y"])
        for key, v in p["out"].items():
            out[f"{prefix}/pipe{k}/{key}"] = np.asarray(v, float)
    for vi, v in enumerate(r.vertices):
        for key, arr in v.items():
            out[f"{prefix}/vertex{vi}/{key}"] = np.asarray(arr, float)
    for kind, d in r.elements.items():
        for vertex, dd in d.items():
            for key, arr in dd.items():
                out[f"{prefix}/{kind}/{vertex}/{key}"] = np.asarray(arr, float)
    return out


def gen_results():
    """Pós-processamento (Results, mass_balance) em 3 instantes para 4 casos, a partir das
    trajetórias golden (capítulo 10) e da referência numpy 150 s."""
    from mfsim.analysis import mass_balance
    from mfsim.cases import chapter10_case, legacy_results
    from mfsim.results import Results
    from mfsim.simulation import build_model

    out = {}
    g = np.load(GOLDEN / "chapter10.npz")
    for key in ("10.1.1", "10.3.1", "10.3.2b"):
        case, raw = chapter10_case(key), legacy_results(key)
        model = build_model(case)
        T, X = g[f"{key}/T"], g[f"{key}/X"]
        idx = np.array([0, T.size // 2, T.size - 1])
        res = Results(model, T, X)
        out.update(_results_arrays(key, res, idx))
        if key == "10.1.1":
            for ph in ("L", "G"):
                m, net, err = mass_balance(res, 0, ph)
                out[f"{key}/mass_balance_{ph}"] = np.stack([np.asarray(m, float), np.asarray(net, float)])
    s = np.load(GOLDEN / "severe.npz")
    run, case, opts = _severe_case(REFERENCE_RUN, tf=150.0, dt_out=2.0)
    model = build_model(case)
    T, X = s["reference_numpy_150s/T"], s["reference_numpy_150s/X"]
    res = Results(model, T, X)
    out.update(_results_arrays("severe150", res, np.array([0, T.size // 2, T.size - 1])))
    for ph in ("L", "G"):
        m, net, err = mass_balance(res, 0, ph)
        out[f"severe150/mass_balance_{ph}"] = np.stack([np.asarray(m, float), np.asarray(net, float)])
    print("  results:", len(out), "arrays")
    return {"results.npz": out}


def _nb_loader_verbatim(mb, T, fluid, D, P_out):
    """Cópia literal da construção do caso em `load_blocked_run` (notebooks_src/nb_severe_slug.py).
    Serve para provar que mfsim.runs.case_from_meta reproduz o notebook; apaga-se quando
    o notebook passar a usar o carregador único."""
    from mfsim.cases.severe_slug import severe_slug_case
    from mfsim.fluid import Fluid

    fluid_b = Fluid(
        rhoL0=mb.get("rhoL", fluid.rhoL0),
        rhoG0=mb.get("rhoG0", fluid.rhoG0),
        muL=mb.get("muL", fluid.muL),
        muG=mb.get("muG", fluid.muG),
        gammaL=mb.get("gammaL", fluid.gammaL),
        gammaG=mb.get("gammaG", fluid.gammaG),
        P0=mb.get("P0", fluid.P0),
        sigma=mb.get("sigma", fluid.sigma),
    )
    D_b = mb.get("D", D)
    A_b = np.pi * D_b**2 / 4
    P_out_b = mb.get("P_out", P_out)
    P_gas = mb.get("P_gas_ref") or (P_out_b + 1.0)
    qL_b = mb.get("vSL", 0.3) * fluid_b.rhoL0 * A_b
    qG_b = mb.get("vSG", 0.1) * fluid_b.rhoG(P_gas * 1e5) * A_b
    return severe_slug_case(
        method="fv_upwind",
        tf=T[-1],
        dt_out=mb.get("dt_out", 2.0),
        qL=qL_b,
        qG=qG_b,
        D=D_b,
        P_out_bar=P_out_b,
        L_h1=mb.get("L_h1", 20.0),
        L_down=mb.get("L_down", 300.0),
        slope_down_deg=mb.get("slope", -3.0),
        H_riser=mb.get("H_riser", 20.0),
        L_h2=mb.get("L_h2", 5.0),
        n_h1=mb.get("n_h1", 2),
        n_down=mb["n_down"],
        n_riser=mb["n_riser"],
        n_h2=2,
        regime=mb.get("regime", 4),
        single_pipe=not mb.get("network", False),
        fluid=fluid_b,
        roughness=mb.get("roughness", 1e-5),
        start=mb.get("start", "blocked"),
        t_ramp=mb.get("t_ramp", 100.0),
        dP_margin_bar=mb.get("dP_margin", 0.3),
        elbow_radius=mb.get("elbow_radius", 0.0),
        n_elbow=mb.get("n_elbow", 0),
        grading_down=mb.get("grading_down", 1.0),
        grading_riser=mb.get("grading_riser", 1.0),
        upstream_volume=mb.get("upstream_volume", 0.0),
        extra={
            "regime_vertical": mb.get("regime_vertical", 0 if mb.get("regime", 4) == 6 else 2),
            "k_switch": mb.get("k_switch", 1e6),
            "ipe_smooth": mb.get("ipe_smooth", False),
            "ipe_local": mb.get("ipe_local", False),
            "alpha_face": mb.get("alpha_face", "upwind"),
            "relax_target": mb.get("relax_target", "raw"),
            "fi_mult": mb.get("fi_mult", 1.0),
            "fi_vgt": mb.get("fi_vgt", 5.5),
            "virtual_mass": mb.get("virtual_mass", 0.0),
            "vm_lambda": mb.get("vm_lambda", 1.0),
            "bubble_drag": mb.get("bubble_drag", "tese"),
            "q_switch_L": 0.02 * qL_b,
            "q_switch_G": 0.02 * qG_b,
            "switch_frac": mb["switch_frac"],
            "relax_time": mb["relax_time"],
            "conservative": mb["conservative"],
            "vertical_blend": mb.get("vertical_blend", 0.0),
            "art_visc": mb.get("art_visc", 0.0),
            "low_void_regime": mb.get("low_void_regime", 0),
            "interfacial_pressure": mb.get("interfacial_pressure", 0.0),
            "high_void_regime": mb.get("high_void_regime", 0),
            "high_void_alpha": mb.get("high_void_alpha", 0.9),
            "vertex_volume": mb.get("vertex_volume", 0.05),
        },
    )


def run_meta_probe(model, T, X):
    """Sondas de uma corrida reconstruída: tamanho do estado, rhs e P_bar no 1.º instante, fluxos."""
    from mfsim.results import Results

    out = {"n_states": np.array(model.n_states), "rhs0": model.rhs(T[0], X[:, 0])}
    r = Results(model, T[:1], X[:, :1])
    out["P_bar0"] = np.asarray(r.pipe(0)["out"]["P_bar"][:, 0], float)
    out["alphaG0"] = np.asarray(r.pipe(0)["out"]["alphaG"][:, 0], float)
    pb = model.pipes[0]
    if getattr(pb.scheme, "conservative", False):
        FL, FG = pb.scheme.mass_fluxes_nodal(r.pipe(0)["Y"][0])
        out["FL0"] = np.asarray(FL, float)
        out["FG0"] = np.asarray(FG, float)
    return out


def gen_runs_meta():
    """Para cada npz de notebooks/data, sondas do modelo reconstruído pela cópia literal do
    carregador do notebook (defaults: fluido do notebook, D 0.05, P_out 4 bar)."""
    import json as _json
    from mfsim.fluid import Fluid
    from mfsim.simulation import build_model

    fluid = Fluid(rhoL0=900.0, rhoG0=1.2, muL=2e-3, muG=1.8e-5, gammaL=100.0, gammaG=1.05, P0=1e5, sigma=0.05)
    out, names = {}, []
    for f in sorted(DATA.glob("*.npz")):
        with np.load(f, allow_pickle=False) as d:
            if "case_json" in d.files:
                mb = dict(_json.loads(str(d["case_json"])).get("meta", {}).get("generator_params") or {})
            elif "meta" in d.files:
                mb = _json.loads(str(d["meta"]))
            else:
                continue
            T = d["T"]
            X = d["X"]
        if any(k not in mb for k in ("n_down", "n_riser", "switch_frac", "relax_time", "conservative")):
            continue  # ficheiros antigos que o notebook lê de outra forma (só T, X)
        case = _nb_loader_verbatim(mb, T, fluid, 0.05, 4.0)
        model = build_model(case)
        for k, v in run_meta_probe(model, T, X).items():
            out[f"{f.stem}/{k}"] = v
        names.append(f.stem)
    out["names"] = np.array(names)
    print("  runs_meta:", len(names), "corridas")
    return {"runs_meta.npz": out}


def gen_nightly():
    """Nível noturno: referência 3000 s em JAX e gas lift 2000 s (duas aberturas)."""
    from mfsim.cases import chapter10_case, legacy_results
    from mfsim.integrators.implicit import ImplicitBDF
    from mfsim.network.graph import Choke
    from mfsim.network.schedules import PiecewiseLinear
    from mfsim.simulation import build_model, simulate, state_bounds

    out = {}
    run, case = _severe_case(REFERENCE_RUN, tf=3000.0, dt_out=2.0)
    t0 = time.time()
    res = simulate(case, backend="jax", progress=False)
    model, T, X, info = res.model, res.T, res.X, res.info
    out["reference_jax_3000s/T"] = T
    out["reference_jax_3000s/X"] = X
    out["reference_jax_3000s/n_steps"] = np.array(info["n_steps"])
    print(f"  referência JAX 3000 s: {info['n_steps']} passos {time.time()-t0:.1f} s")
    for opening in (0.12, 0.10):
        tf = 2000.0
        case, raw = chapter10_case("10.3.2b"), legacy_results("10.3.2b")
        case.set_time(tf=tf)
        for v in case.edges_of(Choke):
            if v.v_to == "6":
                v.opening = PiecewiseLinear(np.array([0.0, 100.0, 150.0, tf]), np.array([0.75, 0.75, opening, opening]))
        model = build_model(case)
        bdf = ImplicitBDF(
            model.rhs,
            S=model.jac_sparsity(),
            rtol=1e-3,
            atol=1e-5,
            bounds=state_bounds(model),
            min_step=1e-9,
            max_order=2,
        )
        T, X, info = bdf.integrate(
            np.arange(0, tf + 1, 5.0), model.initial_state(), breakpoints=model.spec_breakpoints()
        )
        tag = f"gaslift_{int(opening*100)}"
        out[f"{tag}/T"] = T
        out[f"{tag}/X"] = X
        out[f"{tag}/n_steps"] = np.array(info["n_steps"])
        print(f"  gas lift {opening:.2f}: {info['n_steps']} passos")
    return {"nightly.npz": out}


def gen_notebook_text():
    """Outputs de texto (stdout das células) dos notebooks já executados em notebooks/*.ipynb,
    um ficheiro por notebook em tests/golden/notebook_text/ (o teste noturno re-executa e compara)."""
    import json as _json

    nbdir = ROOT / "notebooks"
    outdir = GOLDEN / "notebook_text"
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for nb in sorted(nbdir.glob("*.ipynb")):
        text = notebook_text(nb)
        (outdir / f"{nb.stem}.txt").write_text(text)
        n += 1
    print(f"  notebook_text: {n} notebooks")
    return {}


_VOLATILE = re.compile(r"\b\d+(?:\.\d+)?\s?(ms|s)\b")
_VOLATILE_CPU = re.compile(r"'cpu': \d+(?:\.\d+)?(?:e-?\d+)?")


def normalize_text(text: str) -> str:
    """Substitui os tokens voláteis dos outputs (tempos de CPU/parede como '0.8s', '2.31 s',
    '12 ms', ``'cpu': 0.94`` nos dicionários `info`) por um marcador, para que a comparação
    seja de resultados e não de cronómetro."""
    return _VOLATILE_CPU.sub("'cpu': <t>", _VOLATILE.sub(r"<t>\1", text))


def notebook_text(path) -> str:
    """Concatena os outputs de texto (stream) das células de código de um .ipynb, com os
    tempos de execução normalizados (ver `normalize_text`)."""
    import json as _json

    nb = _json.load(open(path))
    parts = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        # o Jupyter pode partir o stdout de uma célula em vários outputs "stream": um cabeçalho por célula
        text = "".join("".join(o.get("text", [])) for o in c.get("outputs", []) if o.get("output_type") == "stream")
        if text:
            parts.append(f"### célula {i}\n" + text)
    return normalize_text("\n".join(parts))


def _network_arrays(prefix, model, t):
    """Especificações, diagnósticos dos elementos, rhs, layout e rotinas de apoio de um modelo
    (nível A: a montagem da rede sobre o grafo tem de reproduzir isto bit a bit)."""
    X0 = model.initial_state()
    E, info = model.specifications(t, X0)
    out = {
        f"{prefix}/X0": X0,
        f"{prefix}/E": np.asarray(E, float),
        f"{prefix}/rhs": model.rhs(t, X0),
        f"{prefix}/rhs_pert": model.rhs(t, X0 * (1 + 0.02 * np.sin(np.arange(X0.size)))),
        f"{prefix}/state_scales": model.state_scales(X0),
        f"{prefix}/state_slots": model.state_slots(),
        f"{prefix}/sparsity": np.asarray(model.jac_sparsity(), dtype=float).astype(np.int8),
        f"{prefix}/breakpoints": model.spec_breakpoints(),
        f"{prefix}/boundary_net_fluxes": np.asarray(model.boundary_net_fluxes(t, X0), float),
        f"{prefix}/E_t": np.asarray(model.time_inputs(t)[0], float),
        f"{prefix}/opt_t": np.asarray(model.time_inputs(t)[1], float),
    }
    for grp in ("valves", "reservoirs", "check_valves"):
        rows = info.get(grp, [])
        if rows:
            keys = sorted(k for k in rows[0] if k != "vertex")
            out[f"{prefix}/info_{grp}"] = np.array([[float(r[k]) for k in keys] for r in rows], float)
            out[f"{prefix}/info_{grp}_vertex"] = np.array([int(r["vertex"]) for r in rows])
    if model.conservative:
        out[f"{prefix}/transform"] = model.transform(X0)
        out[f"{prefix}/transform_jac"] = model.transform_jacobian(X0).toarray()
    return out


def gen_network():
    """Montagem da rede: todos os casos do capítulo 10 (com variantes de esquema no 10.1.1 e 10.3.1),
    os casos JSON do severe slug e duas corridas em rede."""
    import copy
    from mfsim.cases import chapter10_keys, chapter10_case, legacy_results
    from mfsim.io.spec import case_from_json
    from mfsim.simulation import build_model

    out = {}
    for key in chapter10_keys():
        case = chapter10_case(key)
        t = case.time.t0 + 1.0
        out.update(_network_arrays(f"ch10/{key}", build_model(case), t))
    for key, methods in (("10.1.1", ("fd_central", "fd_upwind", "petrov_galerkin")), ("10.3.1", ("fv_upwind",))):
        for m in methods:
            case = chapter10_case(key)
            c = copy.deepcopy(case)
            c.set_method(m)
            if m == "fv_upwind":
                c.set_options(conservative=True)
            out.update(_network_arrays(f"ch10/{key}_{m}", build_model(c), c.time.t0 + 1.0))
    for f in sorted((ROOT / "cases" / "severe_slug").glob("*.json")):
        case = case_from_json(str(f))
        out.update(_network_arrays(f"severe/{f.stem}", build_model(case), 1.0))
    from mfsim.runs import load_run

    for name in ("severe_slug_blocked_network_v001", "severe_slug_blocked_conservative_arc_tol_hv_vm_lv_bd_phys"):
        p = DATA / f"{name}.npz"
        if p.exists():
            run = load_run(str(p))
            out.update(_network_arrays(f"runs/{name}", run.model, 1.0))
    return {"network.npz": out}


GENERATORS = dict(
    network=gen_network,
    notebook_text=gen_notebook_text,
    chapter10=gen_chapter10,
    variants=gen_variants,
    severe=gen_severe,
    results=gen_results,
    runs_meta=gen_runs_meta,
    nightly=gen_nightly,
)


# ----------------------------------------------------------------------------- escrita
def write_group(name, files, rebaseline, note):
    hashes = json.load(open(GOLDEN / "HASHES.json")) if (GOLDEN / "HASHES.json").exists() else {}
    policy = json.load(open(GOLDEN / "policy.json")) if (GOLDEN / "policy.json").exists() else {}
    if not files:
        return
    for fname, arrays in files.items():
        path = GOLDEN / fname
        max_dev = None
        if path.exists():
            if not rebaseline:
                raise SystemExit(f"{path} existe: use --rebaseline --note '...' para regenerar")
            old = np.load(path)
            devs = []
            for k, v in arrays.items():
                if k in old.files and old[k].shape == np.asarray(v).shape and np.asarray(v).dtype.kind in "fi":
                    a, b = np.asarray(v, float), np.asarray(old[k], float)
                    devs.append(float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b))))))
            max_dev = max(devs) if devs else 0.0
            old.close()
        np.savez(path, **arrays)
        hashes[fname] = {k: array_hash(v) for k, v in arrays.items()}
        for k in arrays:
            policy.setdefault(f"{fname}:{k}", "A")
        with open(GOLDEN / "CHANGELOG.md", "a") as f:
            f.write(
                f"- {_dt.date.today().isoformat()} {git_sha()} `{fname}` {'regenerado' if max_dev is not None else 'criado'}"
                f"{'' if max_dev is None else f' (desvio máx. relativo {max_dev:.2e})'}: {note or 'geração inicial'}\n"
            )
        print(
            f"  gravado {fname}: {len(arrays)} arrays" + (f", desvio máx {max_dev:.2e}" if max_dev is not None else "")
        )
    json.dump(hashes, open(GOLDEN / "HASHES.json", "w"), indent=1, sort_keys=True)
    json.dump(policy, open(GOLDEN / "policy.json", "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--rebaseline", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    groups = [g for g in GROUPS if g != "nightly"] if a.all else a.only
    if not groups:
        ap.error("indique --all ou --only <grupos>")
    GOLDEN.mkdir(parents=True, exist_ok=True)
    if not (GOLDEN / "CHANGELOG.md").exists():
        (GOLDEN / "CHANGELOG.md").write_text("# Registo dos ficheiros golden\n\n")
    prov = provenance()
    print("proveniência:", prov)
    json.dump(prov, open(GOLDEN / "provenance.json", "w"), indent=1)
    for g in groups:
        print(f"[{g}]")
        t0 = time.time()
        write_group(g, GENERATORS[g](), a.rebaseline, a.note)
        print(f"  {time.time()-t0:.0f} s")
