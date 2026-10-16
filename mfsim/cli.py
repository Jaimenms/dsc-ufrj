"""Linha de comando do mfsim: `mfsim run|batch|steady|stats|scan|riser|val|validate|nb`.

    mfsim run cases/severe_slug/reference.json --backend jax --tf 150 --out runs/ref150.npz
    mfsim run cases/severe_slug/reference.json --set integration.integrator=radau --set integration.rtol=1e-5
    mfsim batch batches/thesis.json                               # lote da tese -> runs/chapter10/*.npz
    mfsim steady cases/chapter10/10_1_1.json --out /tmp/10_1_1_ss.json
    mfsim stats severe_slug_blocked_conservative_arc ...        (nomes em notebooks/data ou caminhos .npz)
    mfsim scan severe_slug_blocked_conservative_arc_scan_P2_g005 ...   (varredura da fronteira: Bøe, padrão da linha, ciclos)
    mfsim riser severe_slug_blocked_conservative_arc_tol_hv_vm_lv_bd_phys   (filme no riser: contracorrente)
    mfsim val severe_slug_lab_fabre2_fisico --plot              (validação de laboratório contra os valores publicados)
    mfsim validate cases/**/*.json                              (schema mfsim-case/1)
    mfsim nb notebooks_src/nb_severe_slug.py [--execute]

As corridas são gravadas no formato v2 (`mfsim.runs.save_run`: trajetória, `info_json`,
`provenance_json` e o caso completo em `case_json`); uma falha do integrador grava a
trajetória parcial em `<out>` com `info_json["success"] = false`. `tools/convert_runs.py` traz os
ficheiros antigos (`meta` v1) para este formato; `generators/severe_slug.py` gera casos novos.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def cmd_run(a):
    from .io.spec import case_from_json
    from .runs import save_run
    from .simulation import simulate

    case = case_from_json(a.case, overrides=a.set)
    if a.tf is not None:
        case.set_time(tf=float(a.tf))
    if a.dt_out is not None:
        case.set_time(dt_out=float(a.dt_out))
    overrides = {
        k: getattr(a, k)
        for k in ("rtol", "tol_range", "max_newton", "newton_tol", "first_step")
        if getattr(a, k) is not None
    }
    if a.backend:
        overrides["backend"] = a.backend
    if a.slow is not None:
        overrides["slow_exit"] = bool(a.slow)
    if overrides:
        case.set_integration(**overrides)  # o caso gravado com a corrida reflete as opções efetivas
    it = case.integration
    out = a.out or os.path.join("runs", f"{case.name}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    print(
        f"caso {case.name}: tf = {case.time.tf} s, dt_out = {case.time.dt_out} s, integrador = {it.integrator}, backend = {it.backend}, "
        f"rtol = {it.rtol}, tol_range = {it.tol_range}, max_newton = {it.max_newton}, slow_exit = {it.slow_exit}, estado inicial = {case.initial}",
        flush=True,
    )
    t0 = time.time()
    res = simulate(case, progress=False, verbose=a.verbose, partial=True)
    info = res.info
    T, X = res.T, res.X
    if not info.get("success", True):
        print(
            f"FALHA: {info.get('error')} após {info.get('n_steps')} passos ({time.time() - t0:.1f} s); trajetória parcial até t = {T[-1] if T.size else float('nan')} s",
            flush=True,
        )
    save_run(out, T, X, res.model.network, info=info, acc=info.get("accum"), command=" ".join(sys.argv))
    print(
        f"{'OK' if info.get('success', True) else 'parcial'}: {info.get('n_steps')} passos, {info.get('n_rejected')} rejeitados, "
        f"{info.get('n_rhs')} rhs, {info.get('n_jac')} jac, {time.time() - t0:.1f} s -> {out}",
        flush=True,
    )
    return 0 if info.get("success", True) else 1


def cmd_validate(a):
    from .io.spec import load_json
    from .io.validate import case_errors

    rc = 0
    for f in a.files:
        errs = case_errors(load_json(f))
        print(f"{f}: {'OK' if not errs else str(len(errs)) + ' erro(s)'}", flush=True)
        for e in errs:
            print("   ", e)
        rc = rc or (1 if errs else 0)
    return rc


def cmd_batch(a):
    from .runs import run_batch

    done = run_batch(a.spec, force=a.force, only=a.only or None, jobs=a.jobs)
    return 0 if all(info.get("success", True) for _, _, info in done) else 1


def cmd_steady(a):
    from .io.spec import case_from_json, case_to_json
    from .simulation import steady_state

    case = case_from_json(a.case, overrides=a.set)
    model, X, info = steady_state(case, verbose=a.verbose, jax=(a.backend == "jax"))
    out = a.out or os.path.splitext(a.case)[0] + "_ss.json"
    case_to_json(case, out)
    print(f"estacionário: resíduo {info.get('residual', float('nan')):.3e} -> {out}")
    return 0


def cmd_stats(a):
    from .analysis import CYCLE_STATS_HEADER, cycle_stats, format_cycle_stats
    from .runs import load_run

    print(CYCLE_STATS_HEADER)
    for name in a.runs:
        run = load_run(name)
        st = cycle_stats(run, t_skip=a.t_skip, min_gap=a.min_gap)
        print(format_cycle_stats(os.path.splitext(os.path.basename(name))[0], st), flush=True)
    return 0


def cmd_scan(a):
    from .analysis import SCAN_HEADER, format_scan, scan_summary
    from .runs import load_run

    print(SCAN_HEADER)
    for name in a.runs:
        print(
            format_scan(os.path.splitext(os.path.basename(name))[0], scan_summary(load_run(name), t_skip=a.t_skip)),
            flush=True,
        )
    return 0


def cmd_riser(a):
    from .analysis import format_riser, riser_countercurrent
    from .runs import load_run

    for name in a.runs:
        print(
            format_riser(os.path.splitext(os.path.basename(name))[0], riser_countercurrent(load_run(name))), flush=True
        )
    return 0


def cmd_val(a):
    from .analysis import format_lab, lab_validation
    from .runs import load_run

    rc = 0
    for tag in a.runs:
        name = os.path.splitext(os.path.basename(tag))[0]
        try:
            run = load_run(tag)
            print(format_lab(lab_validation(run, name)), flush=True)
            if a.plot:
                import matplotlib

                matplotlib.use("Agg")
                from .plotting import lab_validation_figure

                os.makedirs(a.out_dir, exist_ok=True)
                out = os.path.join(a.out_dir, f"val_{name}.png")
                lab_validation_figure(run, name, t0=a.t0, t1=a.t1, out=out)
                print("figura:", out, flush=True)
        except Exception as e:  # uma corrida em falta não impede as outras
            print(name, "erro:", e, flush=True)
            rc = 1
    return rc


def cmd_nb(a):
    cmd = [sys.executable, str(ROOT / "tools" / "make_nb.py"), a.src] + (["--execute"] if a.execute else [])
    return subprocess.run(cmd).returncode


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="mfsim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="integra um caso JSON e grava a corrida (npz v2)")
    r.add_argument("case")
    r.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="CAMINHO=VALOR",
        help="altera qualquer campo do caso (JSON), p. ex. integration.integrator=radau, network.edges.line.D=0.06",
    )
    r.add_argument("--out")
    r.add_argument("--backend", choices=["numpy", "jax"])
    r.add_argument("--tf", type=float)
    r.add_argument("--dt-out", dest="dt_out", type=float)
    r.add_argument(
        "--slow",
        dest="slow",
        action="store_true",
        default=None,
        help="recálculo da Jacobiana dentro do Newton (slow_exit)",
    )
    r.add_argument("--no-slow", dest="slow", action="store_false")
    r.add_argument("--rtol", type=float)
    r.add_argument("--tol-range", dest="tol_range", type=float)
    r.add_argument("--max-newton", dest="max_newton", type=int)
    r.add_argument("--newton-tol", dest="newton_tol", type=float)
    r.add_argument("--first-step", dest="first_step", type=float)
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("batch", help="executa um lote de casos (JSON) e grava as corridas; os notebooks só leem")
    b.add_argument("spec")
    b.add_argument("--force", action="store_true", help="reexecuta mesmo que a saída exista")
    b.add_argument("--only", action="append", default=[], metavar="NOME")
    b.add_argument("--jobs", type=int, default=1, help="corridas em paralelo (processos)")
    b.set_defaults(func=cmd_batch)

    s = sub.add_parser(
        "steady", help="estado estacionário de um caso JSON; grava o caso com a condição inicial atualizada"
    )
    s.add_argument("case")
    s.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="CAMINHO=VALOR",
        help="altera qualquer campo do caso (JSON), p. ex. integration.integrator=radau, network.edges.line.D=0.06",
    )
    s.add_argument("--out")
    s.add_argument("--backend", choices=["numpy", "jax"], default="numpy")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_steady)

    t = sub.add_parser("stats", help="estatísticas de ciclo de corridas gravadas (severe slug)")
    t.add_argument("runs", nargs="+")
    t.add_argument("--t-skip", dest="t_skip", type=float, default=100.0)
    t.add_argument("--min-gap", dest="min_gap", type=float, default=200.0)
    t.set_defaults(func=cmd_stats)

    sc = sub.add_parser(
        "scan",
        help="varredura da fronteira de intermitência severa (Bøe, padrão da linha, ciclos) de corridas gravadas",
    )
    sc.add_argument("runs", nargs="+")
    sc.add_argument("--t-skip", dest="t_skip", type=float, default=300.0)
    sc.set_defaults(func=cmd_scan)

    ri = sub.add_parser("riser", help="filme no riser (contracorrente) de corridas gravadas")
    ri.add_argument("runs", nargs="+")
    ri.set_defaults(func=cmd_riser)

    va = sub.add_parser("val", help="validação de laboratório: período, pressões e pico contra os valores publicados")
    va.add_argument("runs", nargs="+")
    va.add_argument("--plot", action="store_true", help="grava a figura val_<nome>.png em --out-dir")
    va.add_argument("--out-dir", dest="out_dir", default="runs")
    va.add_argument("--t0", type=float)
    va.add_argument("--t1", type=float)
    va.set_defaults(func=cmd_val)

    vs = sub.add_parser("validate", help="valida casos JSON contra o schema mfsim-case/1")
    vs.add_argument("files", nargs="+")
    vs.set_defaults(func=cmd_validate)

    n = sub.add_parser("nb", help="converte (e executa) um notebook de notebooks_src")
    n.add_argument("src")
    n.add_argument("--execute", action="store_true")
    n.set_defaults(func=cmd_nb)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
