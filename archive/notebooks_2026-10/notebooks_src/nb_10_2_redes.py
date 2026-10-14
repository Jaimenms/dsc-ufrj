# %% [markdown]
# # 10.2 Redes de dutos
#
# ## 10.2.1 Dois trechos de dutos em série
# Dois dutos de 500 m (25 elementos cada), cenário de 10.1.1, diferenças finitas
# centrais. Formulação A: trechos ligados por especificação de **conexão**
# (pressão do vértice a montante = pressão do 1º nó do trecho a jusante; vazões do
# vértice a jusante = vazões do último nó do trecho a montante). Formulação B:
# trechos acoplados por **vértice dinâmico** (tipo A). Figuras 10.20 a 10.24 e
# Tabelas 10.2 e 10.3.

# %%
import numpy as np, matplotlib.pyplot as plt
from thesis.notebook_utils import describe, run_case, compare_states, show_pipe, plot_vertices
from mfsim.plotting import compare_runs
from thesis.mf2 import legacy_pipe_dict

describe("10.2.1A")
caseA, rawA, resA = run_case("10.2.1A")
compare_states(resA, rawA)
describe("10.2.1B")
caseB, rawB, resB = run_case("10.2.1B")
compare_states(resB, rawB)

# %% [markdown]
# ### Figuras 10.20-10.23: fração de gás, pressão e vazões nos dois trechos (formulação A)

# %%
for k in (0, 1):
    show_pipe(resA, rawA, k)

# %% [markdown]
# ### Formulação A × formulação B (saída do trecho 2)

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, key in zip(axes.ravel(), ["alphaG", "P_bar", "qL", "qG"]):
    compare_runs(
        {"Formulação A (conexão)": resA, "Formulação B (vértice dinâmico)": resB},
        k=1,
        key=key,
        node=-1 if key != "P_bar" else 0,
        ax=ax,
        legacy=legacy_pipe_dict(rawA, 1) if key != "P_bar" else None,
    )
fig.suptitle("Dois trechos em série: formulações A e B (trecho 2)", y=1.02)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Fechamento do balanço de massa no vértice de acoplamento (Figura 10.24, Tabela 10.3)

# %%
for label, res in (("A", resA), ("B", resB)):
    p1, p2 = res.pipe(0), res.pipe(1)
    dL = p1["out"]["qL"][-1] - p2["out"]["qL"][0]
    dG = p1["out"]["qG"][-1] - p2["out"]["qG"][0]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(res.T, dL, label="líquido")
    ax.plot(res.T, dG, label="gás")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("q_saída(1) - q_entrada(2) (kg/s)")
    ax.set_title(f"Formulação {label}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()
    print(f"Formulação {label} - Tabela 10.3")
    print(f"{'t (s)':>6s} {'qL saída 1':>11s} {'qL entrada 2':>13s} {'qG saída 1':>11s} {'qG entrada 2':>13s}")
    for t in (0, 750, 1000, 2000, 3000):
        it = int(np.argmin(np.abs(res.T - t)))
        print(
            f"{res.T[it]:6.0f} {p1['out']['qL'][-1, it]:11.3f} {p2['out']['qL'][0, it]:13.3f} {p1['out']['qG'][-1, it]:11.3f} {p2['out']['qG'][0, it]:13.3f}"
        )

# %% [markdown]
# ## 10.2.2 Rede com ciclo
# Cinco dutos de 200 m (10 elementos), dutos 4 e 5 com D = 0.05 m, vértices
# dinâmicos nas junções (2, 3 e 5), pressão fixa no vértice 4. Figuras 10.26 a 10.30.

# %%
describe("10.2.2")
case, raw, res = run_case("10.2.2")
compare_states(res, raw)
for k in range(5):
    show_pipe(res, raw, k)
plot_vertices(res, raw)


# %% [markdown]
# ## Lado direito compilado (JAX)
# A mesma simulação com `simulate(..., jax=True)`: lado direito compilado por
# `jax.jit` e Jacobiana exata por diferenciação automática (`mfsim/compiled.py`, o mesmo código numpy com o backend JAX),
# no lugar das diferenças finitas coloridas. O resultado deve coincidir com a versão
# numpy dentro da tolerância do integrador (as sequências de passos podem diferir
# porque a Jacobiana muda o Newton); o ganho de tempo cresce com o número de passos
# e de Jacobianas.

# %%
from thesis.notebook_utils import jax_comparison

res_jax = jax_comparison(case, res, rtol=1e-4, atol=1e-6)
