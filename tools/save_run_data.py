"""Converte uma corrida do driver (`runs/<tag>.npz`, ou `archive/scratch_2026-10/npz/severe_D_<tag>.npz`
com o `log` ao lado em `logs/`) em `notebooks/data/<nome>.npz` com `meta` v1 (e `case_json`).
uso: uv run python tools/save_run_data.py <tag> <nome>"""

import json
import os
import re
import sys

import numpy as np

from mfsim.runs import case_from_severe_opts, find_run, meta_from_severe_opts, save_run

tag, name = sys.argv[1], sys.argv[2]
path = find_run(tag)
with np.load(path, allow_pickle=False) as d:
    opts = json.loads(str(d["opts"])) if "opts" in d.files else None
    meta_in = json.loads(str(d["meta"])) if "meta" in d.files else {}
    info = json.loads(str(d["info_json"])) if "info_json" in d.files else {}
    T, X = d["T"], d["X"]
    acc = d["acc"] if "acc" in d.files else None
if not info:
    # ficheiros antigos: passos e CPU no log do driver
    for log in (
        os.path.splitext(path)[0] + ".log",
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
if opts is not None:
    meta = meta_from_severe_opts(opts, info)
    case = case_from_severe_opts(opts)
else:
    meta = dict(meta_in, **{k: info.get(k, meta_in.get(k)) for k in ("n_steps", "n_rejected", "cpu")})
    case = None
os.makedirs(os.path.join("notebooks", "data"), exist_ok=True)
out = save_run(
    os.path.join("notebooks", "data", f"{name}.npz"),
    T,
    X,
    meta,
    acc=acc,
    info=info,
    command=" ".join(sys.argv),
    case=case,
)
print("saved", out, meta)
