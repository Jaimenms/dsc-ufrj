# %% [markdown]
# # 10.1.4 Escoamento horizontal via modelo a 3 equações (Beggs & Brill)
#
# Mesmo cenário de 10.1.1 ($q_G$ de 0.1 para 0.13 kg/s) com o modelo de mistura baseado nas
# correlações de Beggs & Brill (capítulo 6 da tese). Figura 10.17.
#
# **Nota de reprodução.** O ficheiro original foi gravado com uma versão anterior do atrito de
# Beggs & Brill: o gradiente de pressão gravado (0.084 Pa/m) é cerca de 130 vezes menor do que
# o do código final (`modelo_duto_BB.m`, Churchill com correção bifásica). Aqui o caso corre
# tal e qual, a partir do estado gravado, com o modelo final; por isso o transiente inicial e
# os níveis de pressão diferem do MATLAB ([issue 001](../../docs/issues/001-reproducao-capitulo-10.md), ponto 4).
# %%
import numpy as np
import matplotlib.pyplot as plt
from thesis.notebook_utils import thesis_run, compare_states, show_pipe, plot_elements, plot_vertices
from thesis.mf2 import legacy_pipe_dict
from mfsim.plotting import pipe_profiles, compare_runs
from mfsim.runs import load_run

run, res, raw = thesis_run("10.1.4")
compare_states(res, raw)

# %%
show_pipe(res, raw, 0, title="Figura 10.17 - Modelo a 3 equações (Beggs & Brill); MATLAB 2010 a tracejado")

# %% [markdown]
# ---
# *Notebook de reprodução fiel: nenhuma simulação corre aqui. As corridas vêm do lote
# `batches/thesis.json` (`uv run mfsim batch batches/thesis.json`), com a configuração exportada
# do ficheiro original; as correções e melhorias estão nos notebooks de `notebooks_src/issues/`
# e no índice [docs/issues/INDEX.md](../../docs/issues/INDEX.md).*
