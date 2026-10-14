"""Converte uma corrida do antigo driver (`runs/<tag>.npz` ou `archive/scratch_2026-10/npz/severe_D_<tag>.npz`,
com `opts`; passos e CPU lidos do `log` ao lado em `logs/` quando existe) em `notebooks/data/<nome>.npz`
no formato v2 (`tools/legacy_runs.py: convert_run`).
uso: uv run python tools/save_run_data.py <tag> <nome>"""

import json
import os
import re
import sys

import numpy as np

from legacy_runs import convert_run
from mfsim.runs import find_run

tag, name = sys.argv[1], sys.argv[2]
src = find_run(tag)
# passos e CPU do log do driver (ficheiros antigos não os guardam no npz)
info = {}
for log in (
    os.path.splitext(src)[0] + ".log",
    os.path.join("archive", "scratch_2026-10", "logs", f"severe_D_{tag}.log"),
):
    if os.path.exists(log):
        text = open(log).read()
        for key, pat in (
            ("n_steps", r"'n_steps': (\d+)"),
            ("n_rejected", r"'n_rejected': (\d+)"),
            ("cpu", r"'cpu': ([0-9.]+)"),
        ):
            mm = re.search(pat, text)
            if mm:
                info[key] = float(mm.group(1)) if key == "cpu" else int(mm.group(1))
        break
dst = os.path.join("notebooks", "data", f"{name}.npz")
os.makedirs(os.path.dirname(dst), exist_ok=True)
with np.load(src, allow_pickle=False) as d:
    opts = json.loads(str(d["opts"]))
    arrays = {k: d[k] for k in d.files if k != "opts"}
opts.update({f"_{k}": v for k, v in info.items()})
tmp = dst + ".src.npz"
np.savez(tmp, opts=json.dumps(opts), **arrays)
out = convert_run(tmp, dst)
os.remove(tmp)
print("saved", out, info)
