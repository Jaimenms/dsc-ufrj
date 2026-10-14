# %% [markdown]
# # 10.2.2 Rede com ciclo
#
# Cinco dutos de 200 m (10 elementos), dutos 4 e 5 com D = 0.05 m, vértices dinâmicos nas
# junções (2, 3 e 5), pressão fixa no vértice 4. Figuras 10.26 a 10.30.
# %%
import numpy as np
import matplotlib.pyplot as plt
from thesis.notebook_utils import thesis_run, compare_states, show_pipe, plot_elements, plot_vertices
from thesis.mf2 import legacy_pipe_dict
from mfsim.plotting import pipe_profiles, compare_runs
from mfsim.runs import load_run

run, res, raw = thesis_run("10.2.2")
compare_states(res, raw)

# %%
for k in range(5):
    show_pipe(res, raw, k)
plot_vertices(res, raw)

# %% [markdown]
# ---
# *Notebook de reprodução fiel: nenhuma simulação corre aqui. As corridas vêm do lote
# `batches/thesis.json` (`uv run mfsim batch batches/thesis.json`), com a configuração exportada
# do ficheiro original; as correções e melhorias estão nos notebooks de `notebooks_src/issues/`
# e no índice [docs/issues/INDEX.md](../../docs/issues/INDEX.md).*
