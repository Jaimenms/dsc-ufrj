"""Arquiva a pasta `scratch/` (fase 3 da reorganização): scripts úteis vão para `tools/`,
experiências para `archive/scratch_2026-10/{py,obsolete,sh}`, dados (npz, logs, figuras)
para `archive/scratch_2026-10/{npz,logs,nblogs,cmp,cmp2}`. Nada é apagado: só movido
(`git mv` para os ficheiros versionados). Escreve `INDEX.md` (origem -> destino e
propósito de cada script) e `runs_index.json` (opções de cada npz de corrida).

uso: uv run python tools/archive_scratch.py [--dry-run]
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch"
ARCHIVE = ROOT / "archive" / "scratch_2026-10"

TOOLS = {"severe_D.py": "severe_run.py", "save_run_data.py": "save_run_data.py", "cycle_stats.py": "cycle_stats.py",
         "scan_summary.py": "scan_summary.py", "val_summary.py": "val_summary.py", "val_plot.py": "val_plot.py",
         "riser_cc.py": "riser_cc.py", "analyze_run.py": "analyze_run.py", "compare_network.py": "compare_network.py",
         "run_all_cases.py": "run_all_cases.py", "verify_jax.py": "verify_jax.py", "compare_jax.py": "compare_jax.py",
         "diag_lab.py": "diag_lab.py", "diag_state.py": "diag_state.py", "diag_ipe.py": "diag_ipe.py"}
PY = ["restart_ring.py", "restart_probe.py", "reversal_demo.py", "plug_ring.py", "well_fv.py", "lessons_gaslift.py",
      "lessons_massbal.py", "probe_ctrl.py", "probe_gaslift_jax.py", "probe_jump.py", "probe_newton.py", "newton_probe.py",
      "newton_stats.py", "gl_strong.py", "bb_exact_effect.py", "diag_K7.py", "diagStall.py", "diagM.py", "diag2600.py",
      "cont2600.py", "severe_C.py", "severe_B.py", "severe_B_diag.py", "severe_B_diag2.py", "severe_diag.py", "severe_diag2.py"]
OBSOLETE = ["severe1.py", "severe2.py", "severe3.py", "severe_single.py", "gl_diag.py", "gl_run.py", "gl_verbose.py",
            "cmpA.py", "cmp_10_1_1.py", "cmp_scipy.py", "run_10_1_1.py"]
DATA_DIRS = {".npz": "npz", ".log": "logs", ".txt": "logs", ".png": "logs", ".npy": "logs", ".sh": "sh"}
SUBDIRS = ["nblogs", "cmp", "cmp2"]


def purpose(path: Path) -> str:
    try:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    except Exception:
        doc = None
    return (doc or "").strip().splitlines()[0] if doc else "(sem docstring)"


def tracked(path: Path) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))], cwd=ROOT, capture_output=True)
    return r.returncode == 0


def move(src: Path, dst: Path, dry: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
        return
    if tracked(src):
        subprocess.run(["git", "mv", str(src.relative_to(ROOT)), str(dst.relative_to(ROOT))], cwd=ROOT, check=True)
    else:
        shutil.move(str(src), str(dst))


def main(dry=False):
    if not SCRATCH.exists():
        print("scratch/ não existe; nada a fazer"); return
    rows = []
    for name, new in TOOLS.items():
        src = SCRATCH / name
        if src.exists():
            rows.append((name, f"tools/{new}", purpose(src))); move(src, ROOT / "tools" / new, dry)
    for group, names in (("py", PY), ("obsolete", OBSOLETE)):
        for name in names:
            src = SCRATCH / name
            if src.exists():
                rows.append((name, f"archive/scratch_2026-10/{group}/{name}", purpose(src))); move(src, ARCHIVE / group / name, dry)
    leftovers = sorted(p.name for p in SCRATCH.glob("*.py"))
    for name in leftovers:
        src = SCRATCH / name
        rows.append((name, f"archive/scratch_2026-10/obsolete/{name}", purpose(src) + " [não listado]")); move(src, ARCHIVE / "obsolete" / name, dry)
    data = {}
    for p in sorted(SCRATCH.iterdir()):
        if p.is_file() and p.suffix in DATA_DIRS:
            sub = DATA_DIRS[p.suffix]; data[sub] = data.get(sub, 0) + 1; move(p, ARCHIVE / sub / p.name, dry)
    for sub in SUBDIRS:
        if (SCRATCH / sub).exists():
            n = sum(1 for _ in (SCRATCH / sub).rglob("*") if _.is_file()); data[sub] = n
            move(SCRATCH / sub, ARCHIVE / sub, dry)
    if not dry:
        pc = SCRATCH / "__pycache__"
        if pc.exists():
            shutil.rmtree(pc)            # cache do Python, não são dados
        rest = [p for p in SCRATCH.iterdir()]
        if rest:
            print("ficam em scratch/ (não movidos):", [p.name for p in rest])
        else:
            SCRATCH.rmdir()
        # índice das corridas (opções gravadas em cada npz)
        import numpy as np
        runs = {}
        for p in sorted((ARCHIVE / "npz").glob("*.npz")):
            entry = {"size": p.stat().st_size}
            try:
                with np.load(p, allow_pickle=False) as d:
                    entry["keys"] = list(d.files)
                    if "T" in d.files:
                        T = d["T"]; entry["n_out"] = int(T.size); entry["t_end"] = float(T[-1]) if T.size else None
                    if "X" in d.files:
                        entry["X_shape"] = list(d["X"].shape)
                    if "opts" in d.files:
                        entry["opts"] = json.loads(str(d["opts"]))
                    if "meta" in d.files:
                        entry["meta"] = json.loads(str(d["meta"]))
            except Exception as e:
                entry["error"] = f"{type(e).__name__}: {e}"
            runs[p.name] = entry
        (ARCHIVE / "runs_index.json").write_text(json.dumps(runs, indent=1, ensure_ascii=False))
        lines = ["# Arquivo da pasta scratch (2026-10)", "",
                 "Conteúdo da antiga `scratch/` movido na fase 3 da reorganização (nada apagado; os ficheiros",
                 "versionados foram movidos com `git mv`, os dados com `mv`). Os scripts úteis passaram a `tools/`",
                 "(caminhos atualizados); as experiências ficam em `py/` (referidas no README/notebooks ou com lições",
                 "registadas) e `obsolete/` (versões iniciais superadas). Os dados (`npz/`, `logs/`, `nblogs/`, `cmp*/`)",
                 "estão fora do git e fixados pelo `MANIFEST.sha256`; `runs_index.json` lista as opções de cada npz.",
                 "", "Equivalentes atuais: `scratch/severe_D.py '{opts}' tag` -> `tools/severe_run.py` (mesmas opções) ou",
                 "`mfsim run caso.json` (casos em `cases/severe_slug/*.json`); `scratch/cycle_stats.py` -> `mfsim stats`;",
                 "`val_launch*.sh` -> `mfsim run cases/severe_slug/lab_*.json --backend jax`.", "",
                 "| origem | destino | propósito |", "|---|---|---|"]
        lines += [f"| `scratch/{a}` | `{b}` | {c.replace('|', '/')} |" for a, b, c in rows]
        lines += ["", "| dados | ficheiros |", "|---|---|"] + [f"| `{k}/` | {v} |" for k, v in sorted(data.items())]
        (ARCHIVE / "INDEX.md").write_text("\n".join(lines) + "\n")
    print(f"scripts: {len(rows)}; dados: {data}")


if __name__ == "__main__":
    main(dry="--dry-run" in sys.argv)
