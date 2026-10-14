# %% [markdown]
# # 10.3.3 Poço offshore
#
# Rede com 7 arestas: poço (aresta 1, líquido, 300 m; aresta 2, mistura, 1500 m), linha de
# produção (aresta 3: 3500 m horizontais + trecho descendente de 1000 m com −100 m), riser
# (aresta 4, 1500 m), linha de entrega de gás (5), linha de gás até a choke de gás (6) e anular
# de gás (7). Reservatório a 225 bar, chokes de produção (Cv = 30) e de gás (Cv = 5), conexões
# entre arestas. Modelo a 3 equações (intermitente) nas arestas bifásicas e monofásicos nas
# demais; diferenças finitas centrais. Cenário: choke de gás 100 % → 80 % entre 100 e 200 s.
# Figuras 10.48 a 10.54.
# %%
import numpy as np
import matplotlib.pyplot as plt
from thesis.notebook_utils import thesis_run, compare_states, show_pipe, plot_elements, plot_vertices
from thesis.mf2 import legacy_pipe_dict
from mfsim.plotting import pipe_profiles, compare_runs
from mfsim.runs import load_run

run, res, raw = thesis_run("10.3.3")
compare_states(res, raw)

# %%
plot_elements(res, raw)
names = ["poço (líquido)", "poço (bifásico)", "linha de produção", "riser", "entrega de gás", "linha de gás", "anular"]
for k, nm in enumerate(names):
    show_pipe(res, raw, k, title=f"Aresta {k + 1}: {nm}")
plot_vertices(res, raw, indices=[1, 2, 3, 4, 5, 6, 7, 8])

# %% [markdown]
# ---
# *Notebook de reprodução fiel: nenhuma simulação corre aqui. As corridas vêm do lote
# `batches/thesis.json` (`uv run mfsim batch batches/thesis.json`), com a configuração exportada
# do ficheiro original; as correções e melhorias estão nos notebooks de `notebooks_src/issues/`
# e no índice [docs/issues/INDEX.md](../../docs/issues/INDEX.md).*
