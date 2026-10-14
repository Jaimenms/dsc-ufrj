"""Montagem da rede (especificações E, diagnósticos dos elementos, rhs, layout de estados,
escalas, esparsidade, pontos de quebra, fluxos de contorno, mapa conservativo) bit-idêntica ao
golden em todos os casos do capítulo 10, nos casos JSON do severe slug e em corridas em rede:
a rede montada sobre o grafo (resolução de portas) reproduz a montagem legada."""

import copy
from pathlib import Path

import numpy as np
import pytest

from mfsim.cases import chapter10_keys, chapter10_case
from mfsim.io.spec import case_from_json
from mfsim.simulation import build_model

pytestmark = [pytest.mark.golden]
ROOT = Path(__file__).resolve().parents[1]


def _check(g, prefix, model, t, case=None):
    X0 = model.initial_state()
    g.assert_equal(f"{prefix}/X0", X0)
    E, info = model.specifications(t, X0)
    E = np.asarray(E, float)
    E_gold = g[f"{prefix}/E"]
    if E.shape == E_gold.shape:
        g.assert_equal(f"{prefix}/E", E)
    else:
        # o grafo tem as junções explícitas (linhas a mais): compara o que cada duto lê nas
        # extremidades e as linhas dos tanques com as linhas legadas dos vértices
        # (a ordem legada = vértices não-junção na ordem do grafo; Jc<v>/Jr<v> desdobram v)
        shown = [v for v in model.vertices if v.kind != "junction"]
        pos = {int(v.id): i for i, v in enumerate(shown)}
        legacy = lambda vid: int(vid[2:]) if vid.startswith(("Jc", "Jr")) else int(vid)
        for pb in model.pipes:
            assert np.array_equal(
                E[pb.i_from], E_gold[pos[legacy(pb.pipe.v_from)]]
            ), f"{prefix}: duto {pb.pipe.id} entrada"
            assert np.array_equal(E[pb.i_to], E_gold[pos[legacy(pb.pipe.v_to)]]), f"{prefix}: duto {pb.pipe.id} saída"
        for vi in model.vA:
            assert np.array_equal(
                E[vi], E_gold[pos[int(model.vertices[vi].id)]]
            ), f"{prefix}: tanque {model.vertices[vi].id}"
        assert np.array_equal(np.asarray(model.time_inputs(t)[0], float)[: E_gold.shape[0]], g[f"{prefix}/E_t"])
    g.assert_equal(f"{prefix}/rhs", model.rhs(t, X0))
    g.assert_equal(f"{prefix}/rhs_pert", model.rhs(t, X0 * (1 + 0.02 * np.sin(np.arange(X0.size)))))
    g.assert_equal(f"{prefix}/state_scales", model.state_scales(X0))
    g.assert_equal(f"{prefix}/state_slots", model.state_slots())
    g.assert_equal(f"{prefix}/sparsity", np.asarray(model.jac_sparsity(), dtype=float).astype(np.int8))
    g.assert_equal(f"{prefix}/breakpoints", model.spec_breakpoints())
    g.assert_equal(f"{prefix}/boundary_net_fluxes", np.asarray(model.boundary_net_fluxes(t, X0), float))
    E_t, opt_t = model.time_inputs(t)
    g.assert_equal(f"{prefix}/opt_t", np.asarray(opt_t, float))
    if E.shape == E_gold.shape:
        g.assert_equal(f"{prefix}/E_t", np.asarray(E_t, float))
    for grp in ("valves", "reservoirs", "check_valves"):
        rows = info.get(grp, [])
        if g.has(f"{prefix}/info_{grp}"):
            keys = sorted(k for k in rows[0] if k != "vertex")
            g.assert_equal(f"{prefix}/info_{grp}", np.array([[float(r[k]) for k in keys] for r in rows], float))
            g.assert_equal(f"{prefix}/info_{grp}_vertex", np.array([int(r["vertex"]) for r in rows]))
        else:
            assert not rows, f"{prefix}: diagnósticos {grp} inesperados"
    if g.has(f"{prefix}/transform"):
        g.assert_equal(f"{prefix}/transform", model.transform(X0))
        g.assert_equal(f"{prefix}/transform_jac", model.transform_jacobian(X0).toarray())


@pytest.mark.parametrize("key", chapter10_keys())
def test_chapter10_network_assembly(key, golden):
    g = golden("network.npz")
    case = chapter10_case(key)
    _check(g, f"ch10/{key}", build_model(case), case.time.t0 + 1.0, case)


@pytest.mark.parametrize(
    "key,method",
    [("10.1.1", "fd_central"), ("10.1.1", "fd_upwind"), ("10.1.1", "petrov_galerkin"), ("10.3.1", "fv_upwind")],
)
def test_chapter10_scheme_variants(key, method, golden):
    g = golden("network.npz")
    case = chapter10_case(key)
    c = copy.deepcopy(case).set_method(method)
    if method == "fv_upwind":
        c.set_options(conservative=True)
    _check(g, f"ch10/{key}_{method}", build_model(c), c.time.t0 + 1.0, c)


@pytest.mark.parametrize("stem", sorted(p.stem for p in (ROOT / "cases" / "severe_slug").glob("*.json")))
def test_severe_json_network_assembly(stem, golden):
    g = golden("network.npz")
    case = case_from_json(str(ROOT / "cases" / "severe_slug" / f"{stem}.json"))
    _check(g, f"severe/{stem}", build_model(case), 1.0, case)


@pytest.mark.data
@pytest.mark.parametrize(
    "name", ["severe_slug_blocked_network_v001", "severe_slug_blocked_conservative_arc_tol_hv_vm_lv_bd_phys"]
)
def test_runs_network_assembly(name, golden, data_dir):
    from mfsim.runs import load_run

    g = golden("network.npz")
    if not g.has(f"runs/{name}/X0"):
        pytest.skip("sem golden")
    run = load_run(str(data_dir / f"{name}.npz"))
    _check(g, f"runs/{name}", run.model, 1.0, run.case)
