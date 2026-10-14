# %% [markdown]
# # 10.1.3 Escoamento horizontal com padrão bolhas
#
# Mesmo duto (1000 m, D = 0.1 m, 50 elementos, Galerkin) com fechamento de padrão **bolhas**:
# $q_G = 0.1$ kg/s, $q_L$ de 50 para 55 kg/s em rampa entre 150 e 180 s. Figuras 10.15 e 10.16.
# %%
import numpy as np
import matplotlib.pyplot as plt
from thesis.notebook_utils import thesis_run, compare_states, show_pipe, plot_elements, plot_vertices
from thesis.mf2 import legacy_pipe_dict
from mfsim.plotting import pipe_profiles, compare_runs
from mfsim.runs import load_run

run, res, raw = thesis_run("10.1.3")
compare_states(res, raw)

# %%
fig = pipe_profiles(res, keys=("P_bar", "alphaG"), it=0, legacy=legacy_pipe_dict(raw, 0))
fig.suptitle("Figura 10.15 - Condições iniciais (bolhas)", y=1.03)
plt.show()
show_pipe(res, raw, 0, title="Figura 10.16 - Escoamento horizontal, padrão bolhas")

# %% [markdown]
# ---
# *Notebook de reprodução fiel: nenhuma simulação corre aqui. As corridas vêm do lote
# `batches/thesis.json` (`uv run mfsim batch batches/thesis.json`), com a configuração exportada
# do ficheiro original; as correções e melhorias estão nos notebooks de `notebooks_src/issues/`
# e no índice [docs/issues/INDEX.md](../../docs/issues/INDEX.md).*
