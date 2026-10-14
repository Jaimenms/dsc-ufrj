"""Nível noturno (`uv run pytest -m nightly`): corrida de referência completa (3000 s, JAX),
gas lift com intermitência forte (2000 s), validação com dados de laboratório (assinaturas
das corridas gravadas) e re-execução dos notebooks com comparação dos outputs de texto."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mfsim.analysis import cycle_signature
from mfsim.network.graph import Choke
from mfsim.network.schedules import PiecewiseLinear
from mfsim.runs import case_from_meta, integrate_run, integrator_opts_from_meta, load_run

pytestmark = [pytest.mark.nightly, pytest.mark.golden]
ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RUN = "severe_slug_blocked_conservative_arc_tol_hv_vm_lv_bd_phys"


def test_reference_jax_3000s(golden, data_dir):
    pytest.importorskip("jax")
    g = golden("nightly.npz")
    run = load_run(str(data_dir / f"{REFERENCE_RUN}.npz"))
    case = case_from_meta(run.meta, tf=3000.0, dt_out=2.0)
    model, T, X, info = integrate_run(case, dict(integrator_opts_from_meta(run.meta), jax=True))
    g.assert_equal("reference_jax_3000s/T", T)
    g.assert_equal("reference_jax_3000s/X", X)
    from mfsim.results import Results

    p = Results(model, T, X).pipe(0)
    sig = cycle_signature(T, p["out"]["P_bar"][1], p["out"]["qL"][-1])
    assert sig["n_blowdowns"] == 4 and abs(sig["interval"] - 799.0) < 0.1 * 799.0


@pytest.mark.parametrize("opening", [0.12, 0.10])
def test_gaslift_intermittency_2000s(opening, golden):
    from mfsim.cases import load_chapter10_case
    from mfsim.integrators.implicit import ImplicitBDF
    from mfsim.simulation import build_model, state_bounds

    g = golden("nightly.npz")
    tf = 2000.0
    case, raw = load_chapter10_case("10.3.2b")
    case.set_time(tf=tf)
    for v in case.edges_of(Choke):
        if v.v_to == "6":
            v.opening = PiecewiseLinear(np.array([0.0, 100.0, 150.0, tf]), np.array([0.75, 0.75, opening, opening]))
    model = build_model(case)
    bdf = ImplicitBDF(
        model.rhs, S=model.jac_sparsity(), rtol=1e-3, atol=1e-5, bounds=state_bounds(model), min_step=1e-9, max_order=2
    )
    T, X, info = bdf.integrate(np.arange(0, tf + 1, 5.0), model.initial_state(), breakpoints=model.spec_breakpoints())
    tag = f"gaslift_{int(opening*100)}"
    g.assert_equal(f"{tag}/T", T)
    g.assert_equal(f"{tag}/X", X)
    g.assert_equal(f"{tag}/n_steps", np.array(info["n_steps"]))


# Jansen (1", ciclo irregular com refluxo): o intervalo médio entre blowouts varia 58-90 s em
# 600 s com perturbações de 3 % em rtol ou 10 % no primeiro passo (corrida gravada 69 s; medido
# 46.6 s com choke) -- só o número de blowouts, as pressões e o pico são reprodutíveis
@pytest.mark.parametrize(
    "name,min_gap,tol_interval,tol_n",
    [("severe_slug_lab_fabre2_fisico", 40.0, 0.05, 1), ("severe_slug_lab_jansen_fisico", 15.0, 0.40, 2)],
)
def test_lab_validation_signature(name, min_gap, tol_interval, tol_n, data_dir):
    """Re-corre a validação de laboratório e compara a assinatura do ciclo com a corrida gravada
    (nível C: o resultado publicado na documentação não pode mudar além das tolerâncias)."""
    pytest.importorskip("jax")
    run = load_run(str(data_dir / f"{name}.npz"))
    ref = run.signature(min_gap=min_gap, t_skip=0.2 * run.T[-1])
    case = case_from_meta(run.meta, tf=float(run.T[-1]))
    model, T, X, info = integrate_run(case, run.integrator_opts())
    from mfsim.results import Results

    p = Results(model, T, X).pipe(0)
    sig = cycle_signature(T, p["out"]["P_bar"][1], p["out"]["qL"][-1], min_gap=min_gap, t_skip=0.2 * T[-1])
    assert abs(sig["interval"] - ref["interval"]) <= tol_interval * ref["interval"], (sig, ref)
    assert abs(sig["n_blowdowns"] - ref["n_blowdowns"]) <= tol_n, (sig, ref)
    assert abs(sig["P_max"] - ref["P_max"]) <= 0.02 and abs(sig["P_min"] - ref["P_min"]) <= 0.05
    # o pico de vazão é a grandeza mais sensível (instante do blowout contra a amostragem de
    # saída e sequência de passos): Jansen reproduzido a 20 % na fase 0 com o código intocado
    assert abs(sig["q_peak"] - ref["q_peak"]) <= 0.25 * ref["q_peak"]


@pytest.mark.parametrize("nb", sorted(p.stem for p in (ROOT / "notebooks_src").glob("nb_*.py")))
def test_notebooks_execute_same_text(nb, tmp_path):
    """Converte e executa o notebook num diretório temporário e compara os outputs de texto
    das células com tests/golden/notebook_text/<nb>.txt."""
    sys.path.insert(0, str(ROOT / "tools"))
    from golden_generate import notebook_text

    ref = ROOT / "tests" / "golden" / "notebook_text" / f"{nb}.txt"
    if not ref.exists():
        pytest.skip("sem golden de texto")
    work = tmp_path / "notebooks"
    work.mkdir()
    (work / "data").symlink_to(ROOT / "notebooks" / "data")
    src = tmp_path / "notebooks_src"
    src.mkdir()
    shutil.copy(ROOT / "notebooks_src" / f"{nb}.py", src / f"{nb}.py")
    cmd = [sys.executable, str(ROOT / "tools" / "make_nb.py"), str(src / f"{nb}.py"), "--execute"]
    r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stderr[-2000:]
    out = work / f"{nb}.ipynb"
    assert out.exists(), "notebook não foi escrito"
    assert notebook_text(out) == ref.read_text(), "outputs de texto diferentes do golden"
