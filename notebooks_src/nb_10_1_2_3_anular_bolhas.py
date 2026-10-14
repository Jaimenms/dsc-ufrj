# %% [markdown]
# # 10.1.2 Escoamento horizontal anular e 10.1.3 padrão bolhas
#
# Mesmo duto da seção 10.1.1 (1000 m, D = 0.1 m, 50 elementos, Galerkin) com
# fechamento de padrão **anular** (qL = 0.5 kg/s, qG de 5.0 para 5.5 kg/s) e
# de padrão **bolhas** (qG = 0.1 kg/s, qL de 50 para 55 kg/s). As rampas ocorrem
# entre 150 e 180 s. Figuras reproduzidas: 10.10 a 10.13, 10.15 e 10.16.
#
# Observação: nos arquivos originais destes casos a fração de gás na entrada tem
# condição "conector" sem especificação, que corresponde a gradiente nulo
# (von Neumann); o estado inicial gravado só é estacionário com o método de
# Galerkin, o que indica que foi esse o método usado nas figuras.

# %%
import numpy as np, matplotlib.pyplot as plt
from mfsim.notebook_utils import describe, run_case, compare_states, show_pipe
from mfsim.plotting import pipe_profiles, legacy_pipe_dict

# %% [markdown]
# ## Anular (10.1.2)

# %%
describe("10.1.2")
case, raw, res = run_case("10.1.2")
compare_states(res, raw)

# %%
fig = pipe_profiles(res, keys=("P_bar", "alphaG"), it=0, legacy=legacy_pipe_dict(raw, 0))
fig.suptitle("Figura 10.10 - Condições iniciais (anular)", y=1.03); plt.show()
show_pipe(res, raw, 0)

# %% [markdown]
# ### Sensibilidade de malha (Figuras 10.12 e 10.13)
# A tese observa que, a partir de 25 elementos, as variáveis praticamente coincidem.

# %%
import copy
from mfsim.simulation import steady_state, simulate
from mfsim.plotting import compare_runs
def mesh_runs(case, meshes, method, q0):
    out = {}
    for nel in meshes:
        c = copy.deepcopy(case); c.pipes[0].n_elements = np.array([nel]); c.set_method(method)
        nn = nel + 1
        c.pipes[0].ic = np.stack([np.full(nn, q0[0]), np.full(nn, 1.0), np.full(nn, q0[1]), np.full(nn, q0[2])], axis=1)
        steady_state(c, verbose=False)
        out[f"{nel} elem."] = simulate(c, integrator="implicit", rtol=1e-4, atol=1e-6, progress=False)
    return out
legacy = legacy_pipe_dict(raw, 0)
for method, figname in [("galerkin", "10.12"), ("fd_central", "10.13")]:
    runs = mesh_runs(case, [10, 25, 50, 100], method, (0.94, 0.5, 5.0))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, key in zip(axes.ravel(), ["alphaG", "P_bar", "qL", "qG"]):
        compare_runs(runs, key=key, node=-1 if key != "P_bar" else 0, ax=ax, legacy=legacy if key != "P_bar" else None)
    fig.suptitle(f"Figura {figname} - Sensibilidade de malha, anular ({method})", y=1.02); fig.tight_layout(); plt.show()

# %% [markdown]
# ## Bolhas (10.1.3)

# %%
describe("10.1.3")
case, raw, res = run_case("10.1.3")
compare_states(res, raw)
fig = pipe_profiles(res, keys=("P_bar", "alphaG"), it=0, legacy=legacy_pipe_dict(raw, 0))
fig.suptitle("Figura 10.15 - Condições iniciais (bolhas)", y=1.03); plt.show()
show_pipe(res, raw, 0)


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
