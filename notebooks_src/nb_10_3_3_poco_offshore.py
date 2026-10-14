# %% [markdown]
# # 10.3.3 Poço offshore
#
# Rede com 7 arestas: poço (aresta 1, líquido, 300 m; aresta 2, mistura, 1500 m),
# linha de produção (aresta 3: 3500 m horizontal + trecho descendente de 1000 m
# com −100 m), riser (aresta 4, 1500 m), linha de entrega de gás (5), linha de
# gás até a válvula choke de gás (6) e anular de gás (7). Reservatório a 225 bar,
# chokes de produção (Cv = 30) e de gás (Cv = 5), conexões entre arestas.
# Modelo a 3 equações (intermitente) nas arestas bifásicas e monofásicos nas
# demais; diferenças finitas centrais.
#
# Cenário: choke de gás 100 % → 80 % entre 100 e 200 s. Figuras 10.48 a 10.54.

# %%
import numpy as np, matplotlib.pyplot as plt
from mfsim.notebook_utils import describe, run_case, compare_states, show_pipe, plot_elements, plot_vertices

describe("10.3.3")
case, raw, res = run_case("10.3.3")
compare_states(res, raw)

# %% [markdown]
# ## Figuras 10.48-10.54

# %%
plot_elements(res, raw)
names = ["poço (líquido)", "poço (bifásico)", "linha de produção", "riser", "entrega de gás", "linha de gás", "anular"]
for k, nm in enumerate(names):
    show_pipe(res, raw, k, title=f"Aresta {k + 1}: {nm}")
plot_vertices(res, raw, indices=[1, 2, 3, 4, 5, 6, 7, 8])


# %% [markdown]
# ## Lado direito compilado (JAX)
# A mesma simulação com `simulate(..., jax=True)`: lado direito compilado por
# `jax.jit` e Jacobiana exata por diferenciação automática (`mfsim/compiled.py`, o mesmo código numpy com o backend JAX),
# no lugar das diferenças finitas coloridas. O resultado deve coincidir com a versão
# numpy dentro da tolerância do integrador (as sequências de passos podem diferir
# porque a Jacobiana muda o Newton); o ganho de tempo cresce com o número de passos
# e de Jacobianas.

# %%
from mfsim.notebook_utils import jax_comparison
res_jax = jax_comparison(case, res, rtol=1e-4, atol=1e-6)
