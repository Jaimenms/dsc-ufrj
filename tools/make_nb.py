"""Converte scripts .py com marcadores `# %%` / `# %% [markdown]` em notebooks
.ipynb e (opcionalmente) executa-os com nbconvert.

uso: uv run python tools/make_nb.py notebooks_src/xxx.py [--execute]
"""

import re
import subprocess
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def convert(src: Path, dst: Path):
    text = src.read_text(encoding="utf-8")
    parts = re.split(r"^# %%(.*)$", text, flags=re.M)
    cells = []
    # parts: [preamble, marker1, body1, marker2, body2, ...]
    for i in range(1, len(parts), 2):
        marker, body = parts[i].strip(), parts[i + 1]
        body = body.strip("\n")
        if not body.strip():
            continue
        if "markdown" in marker:
            lines = [re.sub(r"^# ?", "", ln) for ln in body.splitlines()]
            cells.append(new_markdown_cell("\n".join(lines)))
        else:
            cells.append(new_code_cell(body))
    nb = new_notebook(
        cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, dst)
    return dst


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path("notebooks") / (src.stem + ".ipynb")
    convert(src, dst)
    print("notebook:", dst)
    if "--execute" in sys.argv:
        cmd = [
            "uv",
            "run",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=3600",
            str(dst),
        ]
        subprocess.run(cmd, check=True)
        print("executado:", dst)
