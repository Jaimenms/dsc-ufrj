# %% [markdown]
# # 10.3.1 Poço simples
#
# Poço vertical de 2048 m (5 in), modelo de mistura a 3 equações (Beggs & Brill, padrão
# intermitente), vértice de entrada com especificação de **reservatório** (170 bar,
# IPL = 0.71 kg/s/bar, IPG = 0.042 kg/s/bar) e vértice de saída com **válvula choke** (Cv = 30)
# a 10 bar. Diferenças finitas centrais, 25 nós. Cenário: choke 100 % → 25 % em rampa (60–90 s),
# 25 % até 500 s, 25 % → 100 % (500–530 s). Figuras 10.32 a 10.34.
#
# Nota: o ficheiro original gravou o duto com padrão estratificado, mas o estado inicial gravado
# só é estacionário com padrão intermitente, como diz o texto; o caso exportado leva essa
# correção ([issue 001](../../docs/issues/001-reproducao-capitulo-10.md)). A integração original
# usou tolerância relativa 0.1, por isso há diferenças visíveis nos transientes.
# %%
import numpy as np
import matplotlib.pyplot as plt
from thesis.notebook_utils import thesis_run, compare_states, show_pipe, plot_elements, plot_vertices
from thesis.mf2 import legacy_pipe_dict
from mfsim.plotting import pipe_profiles, compare_runs
from mfsim.runs import load_run

run, res, raw = thesis_run("10.3.1")
compare_states(res, raw)

# %% [markdown]
# ## Figura 10.32 - Variáveis do trecho de tubulação

# %%
show_pipe(res, raw, 0)

# %% [markdown]
# ## Figuras 10.33 e 10.34 - Válvula e reservatório

# %%
plot_elements(res, raw)
plot_vertices(res, raw)

# %% [markdown]
# ---
# *Notebook de reprodução fiel: nenhuma simulação corre aqui. As corridas vêm do lote
# `batches/thesis.json` (`uv run mfsim batch batches/thesis.json`), com a configuração exportada
# do ficheiro original; as correções e melhorias estão nos notebooks de `notebooks_src/issues/`
# e no índice [docs/issues/INDEX.md](../../docs/issues/INDEX.md).*
