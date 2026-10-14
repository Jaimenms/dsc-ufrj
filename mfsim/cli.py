"""Linha de comando do mfsim: `mfsim run|steady|stats|scan|riser|val|export-case|convert-runs|nb`.

    mfsim run cases/severe_slug/reference.json --backend jax --tf 150 --out runs/ref150.npz
    mfsim steady cases/chapter10/10_1_1.json --out /tmp/10_1_1_ss.json
    mfsim stats severe_slug_blocked_conservative_arc ...        (nomes em notebooks/data ou caminhos .npz)
    mfsim scan severe_slug_blocked_conservative_arc_scan_P2_g005 ...   (varredura da fronteira: Bøe, padrão da linha, ciclos)
    mfsim riser severe_slug_blocked_conservative_arc_tol_hv_vm_lv_bd_phys   (filme no riser: contracorrente)
    mfsim val severe_slug_lab_fabre2_fisico --plot              (validação de laboratório contra os valores publicados)
    mfsim export-case --chapter10 [--out cases/chapter10]
    mfsim export-case --severe-runs
    mfsim export-case --severe --out novo.json --param D=0.05 --param start=blocked
    mfsim nb notebooks_src/nb_severe_slug.py [--execute]

As corridas são gravadas no formato v2 (`mfsim.runs.save_run`: trajetória, `info_json`,
`provenance_json` e o caso completo em `case_json`); uma falha do integrador grava a
trajetória parcial em `<out>` com `info_json["success"] = false`. `convert-runs` traz os
ficheiros antigos (`meta` v1) para este formato.
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


def _parse_params(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not _:
            raise SystemExit(f"--param espera chave=valor: {it}")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def cmd_run(a):
    from .io.spec import case_from_json
    from .runs import save_run
    from .simulation import simulate

    case = case_from_json(a.case)
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


def cmd_convert_runs(a):
    from .io.legacy_runs import convert_run

    rc = 0
    for src in a.files:
        dst = os.path.join(a.out_dir, os.path.basename(src)) if a.out_dir else None
        try:
            written = convert_run(src, dst)
            print(f"{src}: {'já em v2' if written is None else 'convertido -> ' + written}", flush=True)
        except Exception as e:
            print(f"{src}: NÃO convertido ({e})", flush=True)
            rc = 1
    return rc


def cmd_steady(a):
    from .io.spec import case_from_json, case_to_json
    from .simulation import steady_state

    case = case_from_json(a.case)
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


def cmd_export_case(a):
    if a.chapter10:
        from .io.legacy_mf2 import export_chapter10

        kw = {"out_dir": a.out} if a.out else {}
        for pth in export_chapter10(**kw):
            print("escrito", pth)
        return 0
    if a.severe_runs:
        from .cases.severe_slug import export_run_cases

        for pth in export_run_cases():
            print("escrito", pth)
        return 0
    if a.severe:
        from .cases.severe_slug import generate
        from .io.spec import case_to_json

        if not a.out:
            raise SystemExit("--severe exige --out")
        net = generate(**_parse_params(a.param))
        case_to_json(net, a.out)
        print("escrito", a.out)
        return 0
    if a.mf2:
        from .io.spec import case_to_json
        from .io_legacy import load_case

        if not a.out:
            raise SystemExit("--mf2 exige --out")
        case, raw = load_case(a.mf2)
        case_to_json(case, a.out)
        print("escrito", a.out)
        return 0
    raise SystemExit("indique --chapter10, --severe-runs, --severe ou --mf2")


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

    s = sub.add_parser(
        "steady", help="estado estacionário de um caso JSON; grava o caso com a condição inicial atualizada"
    )
    s.add_argument("case")
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

    e = sub.add_parser("export-case", help="exporta casos para JSON (mfsim-case/1)")
    e.add_argument("--chapter10", action="store_true")
    e.add_argument("--severe-runs", dest="severe_runs", action="store_true")
    e.add_argument("--severe", action="store_true")
    e.add_argument("--mf2")
    e.add_argument("--out")
    e.add_argument("--param", action="append", default=[], help="chave=valor (JSON) de severe_slug_case")
    e.set_defaults(func=cmd_export_case)

    cv = sub.add_parser(
        "convert-runs", help="converte corridas no formato antigo (meta v1 / opts do driver) para v2 (case_json)"
    )
    cv.add_argument("files", nargs="+")
    cv.add_argument(
        "--out-dir", dest="out_dir", help="escreve as cópias convertidas nesta pasta (por omissão substitui no lugar)"
    )
    cv.set_defaults(func=cmd_convert_runs)

    n = sub.add_parser("nb", help="converte (e executa) um notebook de notebooks_src")
    n.add_argument("src")
    n.add_argument("--execute", action="store_true")
    n.set_defaults(func=cmd_nb)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
